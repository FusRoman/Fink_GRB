import warnings
from fink_grb.utils.fun_utils import return_verbose_level

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: F401
import numpy as np
import time
import os
import subprocess
import sys
import healpy as hp

from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, explode, col
from pyspark.sql.types import IntegerType, ArrayType

from fink_utils.science.utils import ang2pix, ra2phi, dec2theta
from fink_utils.spark.partitioning import convert_to_datetime
from fink_utils.broker.sparkUtils import init_sparksession, connect_to_raw_database

import fink_grb
from fink_grb.utils.fun_utils import build_spark_submit, join_post_process
from fink_grb.init import get_config, init_logging


def ztf_grb_filter(spark_ztf):
    """
    filter the ztf alerts by taking cross-match values from ztf.

    Parameters
    ----------
    spark_ztf : spark dataframe
        a spark dataframe containing alerts, this following columns are mandatory and have to be at the candidate level.
            - ssdistnr, distpsnr1, neargaia

    Returns
    -------
    spark_filter : spark dataframe
        filtered alerts

    Examples
    --------
    >>> sparkDF = spark.read.format('parquet').load(alert_data)

    >>> spark_filter = ztf_grb_filter(sparkDF)

    >>> spark_filter.count()
    31
    """
    spark_filter = (
        spark_ztf.filter(
            (spark_ztf.candidate.ssdistnr > 5)
            | (
                spark_ztf.candidate.ssdistnr == -999.0
            )  # distance to nearest known SSO above 30 arcsecond
        )
        .filter(
            (spark_ztf.candidate.distpsnr1 > 2)
            | (
                spark_ztf.candidate.distpsnr1 == -999.0
            )  # distance of closest source from Pan-Starrs 1 catalog above 30 arcsecond
        )
        .filter(
            (spark_ztf.candidate.neargaia > 5)
            | (
                spark_ztf.candidate.neargaia == -999.0
            )  # distance of closest source from Gaia DR1 catalog above 60 arcsecond
        )
    )

    return spark_filter


@pandas_udf(ArrayType(IntegerType()))
def box2pixs(ra, dec, radius, NSIDE):
    """
    Return all the pixels from a healpix map with NSIDE
    overlapping the given area defined by the center ra, dec and the radius.

    Parameters
    ----------
    ra : pd.Series
        right ascension columns
    dec : pd.Series
        declination columns
    radius : pd.Series
        error radius of the high energy events, must be in degrees
    NSIDE : pd.Series
        pixels size of the healpix map

    Return
    ------
    ipix_disc : pd.Series
        columns of array containing all the pixel numbers overlapping the error area.

    Examples
    --------
    >>> spark_grb = spark.read.format('parquet').load(grb_data)
    >>> NSIDE = 4

    >>> spark_grb = spark_grb.withColumn(
    ... "err_degree", spark_grb["err_arcmin"] / 60
    ... )
    >>> spark_grb = spark_grb.withColumn("hpix_circle", box2pixs(
    ...     spark_grb.ra, spark_grb.dec, spark_grb.err_degree, F.lit(NSIDE)
    ... ))

    >>> spark_grb.withColumn("hpix", explode("hpix_circle"))\
            .orderBy("hpix")\
                .select(["triggerId", "hpix"]).head(5)
    [Row(triggerId=683499781, hpix=10), Row(triggerId=683499781, hpix=20), Row(triggerId=683499781, hpix=21), Row(triggerId=683499781, hpix=22), Row(triggerId=683499781, hpix=35)]
    """
    theta, phi = dec2theta(dec.values), ra2phi(ra.values)
    vec = hp.ang2vec(theta, phi)
    ipix_disc = [
        hp.query_disc(nside=n, vec=v, radius=np.radians(r))
        for n, v, r in zip(NSIDE.values, vec, radius.values)
    ]
    return pd.Series(ipix_disc)


def ztf_join_gcn_stream(
    ztf_datapath_prefix,
    gcn_datapath_prefix,
    grb_datapath_prefix,
    night,
    exit_after,
    tinterval,
    logs=False,
):
    """
    Join the ztf alerts stream and the gcn stream to find the counterparts of the gcn alerts
    in the ztf stream.

    Parameters
    ----------
    ztf_datapath_prefix : string
        the prefix path where are stored the ztf alerts.
    gcn_datapath_prefix : string
        the prefix path where are stored the gcn alerts.
    grb_datapath_prefix : string
        the prefix path to save GRB join ZTF outputs.
    night : string
        the processing night
    exit_after : int
        the maximum active time in second of the streaming process
    tinterval : int
        the processing interval time in second between the data batch

    Returns
    -------
    None

    Examples
    --------
    >>> ztf_datatest = "fink_grb/test/test_data/ztf_test/online"
    >>> gcn_datatest = "fink_grb/test/test_data/gcn_test"
    >>> grb_dataoutput = "fink_grb/test/test_output"
    >>> ztf_join_gcn_stream(
    ... ztf_datatest,
    ... gcn_datatest,
    ... grb_dataoutput,
    ... "20190903",
    ... 90,
    ... 5
    ... )

    >>> datatest = pd.read_parquet("fink_grb/test/test_data/grb_join_output.parquet").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datajoin = pd.read_parquet(grb_dataoutput + "/grb/year=2019").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)

    >>> assert_frame_equal(datatest, datajoin, check_dtype=False, check_column_type=False, check_categorical=False)

    >>> shutil.rmtree(grb_dataoutput + "/grb/_spark_metadata")
    >>> shutil.rmtree(grb_dataoutput + "/grb/year=2019")
    >>> shutil.rmtree(grb_dataoutput + "/grb_checkpoint")
    """
    logger = init_logging()
    spark = init_sparksession(
        "science2grb_{}{}{}".format(night[0:4], night[4:6], night[6:8])
    )

    NSIDE = 4

    scidatapath = ztf_datapath_prefix + "/science"

    # connection to the ztf science stream
    df_ztf_stream = connect_to_raw_database(
        scidatapath
        + "/year={}/month={}/day={}".format(night[0:4], night[4:6], night[6:8]),
        scidatapath
        + "/year={}/month={}/day={}".format(night[0:4], night[4:6], night[6:8]),
        latestfirst=False,
    )

    df_ztf_stream = ztf_grb_filter(df_ztf_stream)

    gcn_rawdatapath = gcn_datapath_prefix + "/raw"

    # connection to the gcn stream
    df_grb_stream = connect_to_raw_database(
        gcn_rawdatapath
        + "/year={}/month={}/day={}".format(night[0:4], night[4:6], night[6:8]),
        gcn_rawdatapath
        + "/year={}/month={}/day={}".format(night[0:4], night[4:6], night[6:8]),
        latestfirst=True,
    )

    if logs:  # pragma: no cover
        logger.info("connection to the database successfull")

    # compute healpix column for each streaming df
    df_ztf_stream = df_ztf_stream.withColumn(
        "hpix",
        ang2pix(df_ztf_stream.candidate.ra, df_ztf_stream.candidate.dec, F.lit(NSIDE)),
    )

    df_grb_stream = df_grb_stream.withColumn(
        "err_degree", df_grb_stream["err_arcmin"] / 60
    )
    df_grb_stream = df_grb_stream.withColumn(
        "hpix_circle",
        box2pixs(
            df_grb_stream.ra, df_grb_stream.dec, df_grb_stream.err_degree, F.lit(NSIDE)
        ),
    )
    df_grb_stream = df_grb_stream.withColumn("hpix", explode("hpix_circle"))

    if logs:  # pragma: no cover
        logger.info("Healpix columns computing successfull")

    df_ztf_stream = df_ztf_stream.withColumn("ztf_ra", col("candidate.ra")).withColumn(
        "ztf_dec", col("candidate.dec")
    )

    df_grb_stream = df_grb_stream.withColumnRenamed("ra", "grb_ra").withColumnRenamed(
        "dec", "grb_dec"
    )

    # join the two streams according to the healpix columns.
    # A pixel id will be assign to each alerts / gcn according to their position in the sky.
    # Each alerts / gcn with the same pixel id are in the same area of the sky.
    # The NSIDE correspond to a resolution of ~15 degree/pixel.

    # WARNING  !
    # the join condition with healpix column doesn't work properly
    # have to take into account the nearby pixels in case the error box of a GRB
    # overlap many pixels.
    join_condition = [
        df_ztf_stream.hpix == df_grb_stream.hpix,
        df_ztf_stream.candidate.jdstarthist > df_grb_stream.triggerTimejd,
    ]
    df_grb = df_ztf_stream.join(df_grb_stream, join_condition, "inner")

    df_grb = join_post_process(df_grb)

    # re-create partitioning columns if needed.
    timecol = "jd"
    converter = lambda x: convert_to_datetime(x)  # noqa: E731
    if "timestamp" not in df_grb.columns:
        df_grb = df_grb.withColumn("timestamp", converter(df_grb[timecol]))

    if "year" not in df_grb.columns:
        df_grb = df_grb.withColumn("year", F.date_format("timestamp", "yyyy"))

    if "month" not in df_grb.columns:
        df_grb = df_grb.withColumn("month", F.date_format("timestamp", "MM"))

    if "day" not in df_grb.columns:
        df_grb = df_grb.withColumn("day", F.date_format("timestamp", "dd"))

    grbdatapath = grb_datapath_prefix + "/grb"
    checkpointpath_grb_tmp = grb_datapath_prefix + "/grb_checkpoint"

    query_grb = (
        df_grb.writeStream.outputMode("append")
        .format("parquet")
        .option("checkpointLocation", checkpointpath_grb_tmp)
        .option("path", grbdatapath)
        .partitionBy("year", "month", "day")
        .trigger(processingTime="{} seconds".format(tinterval))
        .start()
    )

    if logs:  # pragma: no cover
        logger.info("Stream launching successfull")

    # Keep the Streaming running until something or someone ends it!
    if exit_after is not None:
        time.sleep(int(exit_after))
        query_grb.stop()
        logger.info("Exiting the science2grb streaming subprocess normally...")
    else:  # pragma: no cover
        # Wait for the end of queries
        spark.streams.awaitAnyTermination()


def launch_joining_stream(arguments):
    """
    Launch the joining stream job.

    Parameters
    ----------
    arguments : dictionnary
        arguments parse by docopt from the command line

    Returns
    -------
    None

    Examples
    --------
    >>> grb_datatest = "fink_grb/test/test_output"
    >>> gcn_datatest = "fink_grb/test/test_data/gcn_test"
    >>> launch_joining_stream({
    ... "--config" : None,
    ... "--night" : "20190903",
    ... "--exit_after" : 90
    ... })

    >>> datatest = pd.read_parquet("fink_grb/test/test_data/grb_join_output.parquet").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datajoin = pd.read_parquet(grb_datatest + "/grb/year=2019").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)

    >>> assert_frame_equal(datatest, datajoin, check_dtype=False, check_column_type=False, check_categorical=False)

    >>> shutil.rmtree(grb_datatest + "/grb/_spark_metadata")
    >>> shutil.rmtree(grb_datatest + "/grb/year=2019")
    >>> shutil.rmtree(grb_datatest + "/grb_checkpoint")
    """
    config = get_config(arguments)
    logger = init_logging()

    verbose = return_verbose_level(config, logger)

    try:
        master_manager = config["STREAM"]["manager"]
        principal_group = config["STREAM"]["principal"]
        secret = config["STREAM"]["secret"]
        role = config["STREAM"]["role"]
        executor_env = config["STREAM"]["exec_env"]
        driver_mem = config["STREAM"]["driver_memory"]
        exec_mem = config["STREAM"]["executor_memory"]
        max_core = config["STREAM"]["max_core"]
        exec_core = config["STREAM"]["executor_core"]

        ztf_datapath_prefix = config["PATH"]["online_ztf_data_prefix"]
        gcn_datapath_prefix = config["PATH"]["online_gcn_data_prefix"]
        grb_datapath_prefix = config["PATH"]["online_grb_data_prefix"]
        tinterval = config["STREAM"]["tinterval"]
    except Exception as e:  # pragma: no cover
        logger.error("Config entry not found \n\t {}".format(e))
        exit(1)

    try:
        night = arguments["--night"]
    except Exception as e:  # pragma: no cover
        logger.error("Command line arguments not found: {}\n{}".format("--night", e))
        exit(1)

    try:
        exit_after = arguments["--exit_after"]
    except Exception as e:  # pragma: no cover
        logger.error(
            "Command line arguments not found: {}\n{}".format("--exit_after", e)
        )
        exit(1)

    try:
        external_python_libs = config["STREAM"]["external_python_libs"]
    except Exception as e:
        if verbose:
            logger.info(
                "No external python dependencies specify in the following config file: {}\n\t{}".format(
                    arguments["--config"], e
                )
            )
        external_python_libs = ""

    application = os.path.join(
        os.path.dirname(fink_grb.__file__),
        "online",
        "ztf_join_gcn.py prod",
    )

    application += " " + ztf_datapath_prefix
    application += " " + gcn_datapath_prefix
    application += " " + grb_datapath_prefix
    application += " " + night
    application += " " + str(exit_after)
    application += " " + tinterval

    spark_submit = "spark-submit \
        --master {} \
        --conf spark.mesos.principal={} \
        --conf spark.mesos.secret={} \
        --conf spark.mesos.role={} \
        --conf spark.executorEnv.HOME={} \
        --driver-memory {}G \
        --executor-memory {}G \
        --conf spark.cores.max={} \
        --conf spark.executor.cores={}".format(
        master_manager,
        principal_group,
        secret,
        role,
        executor_env,
        driver_mem,
        exec_mem,
        max_core,
        exec_core,
    )

    spark_submit = build_spark_submit(
        spark_submit, application, external_python_libs, "", ""
    )

    process = subprocess.Popen(
        spark_submit,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )

    stdout, stderr = process.communicate()
    if process.returncode != 0:  # pragma: no cover
        logger.error(
            "Fink_GRB joining stream spark application has ended with a non-zero returncode.\
                \n\t cause:\n\t\t{}\n\t\t{}".format(
                stdout, stderr
            )
        )
        exit(1)

    logger.info("Fink_GRB joining stream spark application ended normally")
    return


if __name__ == "__main__":

    if sys.argv[1] == "test":
        from fink_utils.test.tester import spark_unit_tests_science
        from pandas.testing import assert_frame_equal  # noqa: F401
        import shutil  # noqa: F401

        globs = globals()

        grb_data = "fink_grb/test/test_data/gcn_test/raw/year=2019/month=09/day=03"
        join_data = "fink_grb/test/test_data/join_raw_datatest.parquet"
        alert_data = "fink_grb/test/test_data/ztf_test/online/science/year=2019/month=09/day=03/ztf_science_test.parquet"
        globs["join_data"] = join_data
        globs["alert_data"] = alert_data
        globs["grb_data"] = grb_data

        # Run the test suite
        spark_unit_tests_science(globs)

    elif sys.argv[1] == "prod":  # pragma: no cover

        ztf_datapath_prefix = sys.argv[2]
        gcn_datapath_prefix = sys.argv[3]
        grb_datapath_prefix = sys.argv[4]
        night = sys.argv[5]
        exit_after = sys.argv[6]
        tinterval = sys.argv[7]

        ztf_join_gcn_stream(
            ztf_datapath_prefix,
            gcn_datapath_prefix,
            grb_datapath_prefix,
            night,
            exit_after,
            tinterval,
        )

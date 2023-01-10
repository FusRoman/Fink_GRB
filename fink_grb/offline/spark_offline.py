import json
from astropy.time import TimeDelta

from fink_utils.science.utils import ang2pix
from fink_utils.broker.sparkUtils import init_sparksession

from pyspark.sql import functions as F
from pyspark.sql.functions import explode, col
import os
import sys
import subprocess
from dateutil import parser

from astropy.time import Time

from fink_utils.spark.partitioning import convert_to_datetime

import fink_grb
from fink_grb.utils.fun_utils import (
    return_verbose_level,
    build_spark_submit,
    join_post_process,
)
from fink_grb.init import get_config, init_logging
from fink_grb.online.ztf_join_gcn import box2pixs


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

    >>> sparkDF = sparkDF.select(
    ... "objectId",
    ... "candid",
    ... "candidate.ra",
    ... "candidate.dec",
    ... "candidate.jd",
    ... "candidate.jdstarthist",
    ... "candidate.jdendhist",
    ... "candidate.ssdistnr",
    ... "candidate.distpsnr1",
    ... "candidate.neargaia",
    ... )

    >>> spark_filter = ztf_grb_filter(sparkDF)

    >>> spark_filter.count()
    31
    """
    spark_filter = (
        spark_ztf.filter(
            (spark_ztf.ssdistnr > 5.0)
            | (
                spark_ztf.ssdistnr == -999.0
            )  # distance to nearest known SSO above 30 arcsecond
        )
        .filter(
            (spark_ztf.distpsnr1 > 2.0)
            | (
                spark_ztf.distpsnr1 == -999.0
            )  # distance of closest source from Pan-Starrs 1 catalog above 30 arcsecond
        )
        .filter(
            (spark_ztf.neargaia > 5.0)
            | (
                spark_ztf.neargaia == -999.0
            )  # distance of closest source from Gaia DR1 catalog above 60 arcsecond
        )
    )

    return spark_filter


def spark_offline(
    hbase_catalog,
    gcn_read_path,
    grbxztf_write_path,
    night,
    start_window,
    time_window,
    with_columns_filter=True,
):
    """
    Cross-match Fink and the GNC in order to find the optical alerts falling in the error box of a GCN.

    Parameters
    ----------
    hbase_catalog : string
        path to the hbase catalog (json format)
        Key index must be jd_objectId
    gcn_read_path : string
        path to the gcn database
    grbxztf_write_path : string
        path to store the cross match ZTF/GCN results
    night : string
        launching night of the script
    start_window : float
        start date of the time window (in jd / julian date)
    time_window : int
        Number of day between start_window and (start_window - time_window) to join ztf alerts and gcn.
        time_window are in days.

    Returns
    -------
    None

    Examples
    --------
    >>> fink_home = os.environ["FINK_HOME"]
    >>> hbase_catalog = fink_home + "/catalogs_hbase/ztf.jd.json"
    >>> gcn_datatest = "fink_grb/test/test_data/gcn_test"
    >>> grb_dataoutput = "fink_grb/test/test_output"
    >>> from astropy.time import Time

    >>> spark_offline(
    ... hbase_catalog,
    ... gcn_datatest,
    ... grb_dataoutput,
    ... "20190903",
    ... Time("2019-09-04").jd,
    ... 7,
    ... False
    ... )

    >>> datatest = pd.read_parquet("fink_grb/test/test_data/grb_join_output.parquet").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datatest = datatest.drop(["delta_mag", "rate", "from_upper", "start_vartime", "diff_vartime", "grb_proba"], axis=1)
    >>> datajoin = pd.read_parquet(grb_dataoutput + "/year=2019").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datajoin = datajoin.drop("grb_proba", axis=1)

    >>> assert_frame_equal(datatest, datajoin, check_dtype=False, check_column_type=False, check_categorical=False)

    >>> shutil.rmtree(grb_dataoutput + "/year=2019")
    >>> os.remove(grb_dataoutput + "/_SUCCESS")
    """
    with open(hbase_catalog) as f:
        catalog = json.load(f)

    spark = init_sparksession(
        "science2grb_offline_{}{}{}".format(night[0:4], night[4:6], night[6:8])
    )

    ztf_alert = (
        spark.read.option("catalog", catalog)
        .format("org.apache.hadoop.hbase.spark")
        .option("hbase.spark.use.hbasecontext", False)
        .option("hbase.spark.pushdown.columnfilter", with_columns_filter)
        .load()
        .filter(~col("jd_objectId").startswith("schema_"))
    )

    ztf_alert = ztf_alert.select(
        "jd_objectId",
        "objectId",
        "candid",
        "ra",
        "dec",
        "jd",
        "fid",
        "rb",
        "jdstarthist",
        "jdendhist",
        "ssdistnr",
        "distpsnr1",
        "neargaia",
        "cdsxmatch",
        "roid",
        "mulens",
        "snn_snia_vs_nonia",
        "snn_sn_vs_all",
        "rf_snia_vs_nonia",
        "ndethist",
        "drb",
        "classtar",
        "rf_kn_vs_nonkn",
        "tracklet",
    )

    low_bound = start_window - TimeDelta(time_window * 24 * 3600, format="sec").jd

    if low_bound < 0 or low_bound > start_window:
        raise ValueError(
            "The time window is higher than the start_window : \nstart_window = {}\ntime_window = {}\nlow_bound={}".format(
                start_window, time_window, low_bound
            )
        )

    ztf_alert = ztf_alert.filter(
        ztf_alert["jd_objectId"] >= "{}".format(low_bound)
    ).filter(ztf_alert["jd_objectId"] < "{}".format(start_window))

    ztf_alert = ztf_grb_filter(ztf_alert)

    ztf_alert.cache().count()

    grb_alert = spark.read.format("parquet").load(gcn_read_path)

    grb_alert = grb_alert.filter(grb_alert.triggerTimejd >= low_bound).filter(
        grb_alert.triggerTimejd <= start_window
    )

    grb_alert.cache().count()

    NSIDE = 4

    ztf_alert = ztf_alert.withColumn(
        "hpix",
        ang2pix(ztf_alert.ra, ztf_alert.dec, F.lit(NSIDE)),
    )

    grb_alert = grb_alert.withColumn("err_degree", grb_alert["err_arcmin"] / 60)
    grb_alert = grb_alert.withColumn(
        "hpix_circle",
        box2pixs(grb_alert.ra, grb_alert.dec, grb_alert.err_degree, F.lit(NSIDE)),
    )
    grb_alert = grb_alert.withColumn("hpix", explode("hpix_circle"))

    ztf_alert = ztf_alert.withColumnRenamed("ra", "ztf_ra").withColumnRenamed(
        "dec", "ztf_dec"
    )

    grb_alert = grb_alert.withColumnRenamed("ra", "grb_ra").withColumnRenamed(
        "dec", "grb_dec"
    )

    join_condition = [
        ztf_alert.hpix == grb_alert.hpix,
        ztf_alert.jdstarthist > grb_alert.triggerTimejd,
        ztf_alert.jdendhist - grb_alert.triggerTimejd <= 10,
    ]
    join_ztf_grb = ztf_alert.join(grb_alert, join_condition, "inner")

    df_grb = join_post_process(join_ztf_grb, with_rate=False, from_hbase=True)

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

    df_grb.write.mode("append").partitionBy("year", "month", "day").parquet(
        grbxztf_write_path
    )


def launch_offline_mode(arguments, is_test=False):
    """
    Launch the offline grb module, used by the command line interface.

    Parameters
    ----------
    arguments : dictionnary
        arguments parse from the command line.

    Returns
    -------
    None

    Examples
    --------
    >>> gcn_datatest = "fink_grb/test/test_data/gcn_test"
    >>> grb_dataoutput = "fink_grb/test/test_output"
    >>> launch_offline_mode({
    ... "--config" : None,
    ... "--night" : "20190903",
    ... "--exit_after" : 90
    ... },
    ... is_test=True
    ... )

    >>> datatest = pd.read_parquet("fink_grb/test/test_data/grb_join_output.parquet").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datatest = datatest.drop(["delta_mag", "rate", "from_upper", "start_vartime", "diff_vartime", "grb_proba"], axis=1)
    >>> datajoin = pd.read_parquet(grb_dataoutput + "/year=2019").sort_values(["objectId", "triggerId", "grb_ra"]).reset_index(drop=True)
    >>> datajoin = datajoin.drop("grb_proba", axis=1)

    >>> assert_frame_equal(datatest, datajoin, check_dtype=False, check_column_type=False, check_categorical=False)

    >>> shutil.rmtree(grb_dataoutput + "/year=2019")
    >>> os.remove(grb_dataoutput + "/_SUCCESS")
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

        gcn_datapath_prefix = config["PATH"]["online_gcn_data_prefix"]
        grb_datapath_prefix = config["PATH"]["online_grb_data_prefix"]
        hbase_catalog = config["PATH"]["hbase_catalog"]
        if is_test:
            try:
                fink_home = os.environ["FINK_HOME"]
                hbase_catalog = fink_home + "/catalogs_hbase/ztf.jd.json"
            except Exception as e:
                logger.error(
                    "FINK_HOME environment variable not found \n\t {}".format(e)
                )

        time_window = int(config["OFFLINE"]["time_window"])
    except Exception as e:  # pragma: no cover
        logger.error("Config entry not found \n\t {}".format(e))
        exit(1)

    try:
        night = arguments["--night"]
    except Exception as e:  # pragma: no cover
        logger.error("Command line arguments not found: {}\n{}".format("--night", e))
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

    try:
        spark_jars = config["STREAM"]["jars"]
        if is_test:
            fink_home = os.environ["FINK_HOME"]
            spark_jars = "{}/libs/fink-broker_2.11-1.2.jar,{}/libs/hbase-spark-hbase2.2_spark3_scala2.11_hadoop2.7.jar,{}/libs/hbase-spark-protocol-shaded-hbase2.2_spark3_scala2.11_hadoop2.7.jar".format(
                fink_home, fink_home, fink_home
            )

    except Exception as e:
        if verbose:
            logger.info(
                "No spark jars dependencies specify in the following config file: {}\n\t{}".format(
                    arguments["--config"], e
                )
            )
        spark_jars = ""

    try:
        packages = config["STREAM"]["packages"]
    except Exception as e:
        if verbose:
            logger.info(
                "No packages dependencies specify in the following config file: {}\n\t{}".format(
                    arguments["--config"], e
                )
            )
        packages = ""

    application = os.path.join(
        os.path.dirname(fink_grb.__file__),
        "offline",
        "spark_offline.py prod",
    )

    start_window_in_jd = Time(parser.parse(night), format="datetime").jd + 0.49

    application += " " + hbase_catalog
    application += " " + gcn_datapath_prefix
    application += " " + grb_datapath_prefix
    application += " " + night
    application += " " + str(start_window_in_jd)
    application += " " + str(time_window)

    if is_test:
        application += " " + str(False)
    else:
        application += " " + str(True)

    spark_submit = "spark-submit \
            --master {} \
            --conf spark.mesos.principal={} \
            --conf spark.mesos.secret={} \
            --conf spark.mesos.role={} \
            --conf spark.executorEnv.HOME={} \
            --driver-memory {}G \
            --executor-memory {}G \
            --conf spark.cores.max={} \
            --conf spark.executor.cores={} \
            ".format(
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
        spark_submit, application, external_python_libs, spark_jars, packages
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
        from fink_utils.test.tester import spark_unit_tests_broker
        from pandas.testing import assert_frame_equal  # noqa: F401
        import shutil  # noqa: F401
        import pandas as pd  # noqa: F401

        globs = globals()

        join_data = "fink_grb/test/test_data/join_raw_datatest.parquet"
        alert_data = "fink_grb/test/test_data/ztf_test/online/science/year=2019/month=09/day=03/ztf_science_test.parquet"
        globs["join_data"] = join_data
        globs["alert_data"] = alert_data

        # Run the test suite
        spark_unit_tests_broker(globs)

    if sys.argv[1] == "prod":  # pragma: no cover

        hbase_catalog = sys.argv[2]
        gcn_datapath_prefix = sys.argv[3]
        grb_datapath_prefix = sys.argv[4]
        night = sys.argv[5]
        start_window = float(sys.argv[6])
        time_window = int(sys.argv[7])
        column_filter = True if sys.argv[8] == "True" else False

        spark_offline(
            hbase_catalog,
            gcn_datapath_prefix,
            grb_datapath_prefix,
            night,
            start_window,
            time_window,
            with_columns_filter=column_filter,
        )

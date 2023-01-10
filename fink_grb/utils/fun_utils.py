import numpy as np
import pandas as pd
from pyarrow import fs

import pyspark.sql.functions as F

from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DoubleType, ArrayType

from fink_filters.classification import extract_fink_classification
from fink_utils.spark.utils import concat_col

from fink_grb.utils.grb_prob import grb_assoc


def return_verbose_level(config, logger):
    """
    Get the verbose level from the config file and return it.

    Parameters
    ----------
    config : dictionnary
        dictionnary containing the key values pair from the config file
    logger : logging object
        the logger used to print logs

    Returns
    -------
    logs : boolean
        if True, print the logs

    Examples
    --------
    >>> c = get_config({"--config" : "fink_grb/conf/fink_grb.conf"})
    >>> logger = init_logging()

    >>> return_verbose_level(c, logger)
    False
    """
    try:
        logs = config["ADMIN"]["verbose"] == "True"
    except Exception as e:
        logger.error(
            "Config entry not found \n\t {}\n\tsetting verbose to True by default".format(
                e
            )
        )
        logs = True

    return logs


def get_hdfs_connector(host: str, port: int, user: str):
    """
    Initialise a connector to HDFS.

    To work, please read :
    https://arrow.apache.org/docs/python/filesystems.html#hadoop-distributed-file-system-hdfs

    The following command line can be used to locate the libhdfs.so file:
        `locate -l 1 libhdfs.so`

    Parameters
    ----------
    host: str
        IP address for the host machine
    port: int
        Port to access HDFS data.
    user: str
        Username on Hadoop.

    Returns
    -------
    fs: pyarrow.hdfs.HadoopFileSystem
    """
    return fs.HadoopFileSystem(host, port, user=user)  # work only for pyarrow >= 2.0.0


def build_spark_submit(
    spark_submit, application, external_python_libs, spark_jars, packages
):
    """
    Build the spark submit command line to launch spark jobs.

    Parameters
    ----------
    spark_submit : string
        Initial spark_submit application containing the options the launch the jobs
    application : string
        The python script and their options that will be launched with the spark jobs
    external_python_libs : string
        list of external python module in .eggs format separated by ','.
    spark_jars : string
        list of external java libraries separated by ','.
    packages : string
        list of external java libraries hosted on maven, the java packages manager.

    Return
    ------
    spark_submit + application : string
        the initial spark_submit string with the additionnal options and libraries add to the spark_submit

    Examples
    --------
    >>> spark_submit = "spark-submit --master local[2] --driver-memory 8G --executor-memory 4G --conf spark.cores.max=4 --conf spark.executor.cores=2"
    >>> application = "myscript.py"
    >>> external_python_libs = "mypythonlibs.eggs,mypythonlibs2.py"
    >>> spark_jars = "myjavalib.jar,myjavalib2.jar"
    >>> packages = "org.apache.mylib:sublib:1.0.0"

    >>> build_spark_submit(spark_submit, application, external_python_libs, spark_jars, packages)
    'spark-submit --master local[2] --driver-memory 8G --executor-memory 4G --conf spark.cores.max=4 --conf spark.executor.cores=2 --py-files mypythonlibs.eggs,mypythonlibs2.py  --jars myjavalib.jar,myjavalib2.jar  --packages org.apache.mylib:sublib:1.0.0  myscript.py'

    >>> build_spark_submit(spark_submit, application, "", "", "")
    'spark-submit --master local[2] --driver-memory 8G --executor-memory 4G --conf spark.cores.max=4 --conf spark.executor.cores=2 myscript.py'
    """

    if application == "":
        raise ValueError("application parameters is empty !!")

    if external_python_libs != "":
        spark_submit += " --py-files {} ".format(external_python_libs)

    if spark_jars != "":
        spark_submit += " --jars {} ".format(spark_jars)

    if packages != "":
        spark_submit += " --packages {} ".format(packages)

    return spark_submit + " " + application


def sub_compute_rate(
    mag, jdstarthist, curr_jd, curr_fid, hist_mag, hist_diffmag, h_jd, h_fid
):
    """
    Compute the rate between the current magnitude measurement in the alerts
    and the last measurement contains in the history.
    If the last measurement doesn't exists, take the last upper limit.

    Parameters
    ----------
    mag : pd.Series
        magnitude estimation of the current alerts
    jdstarthist : pd.Series
        earliest julian date corresponding to the start variation time of the object at 3 sigma.
    curr_jd : pd.Series
        current julian date
    curr_fid : pd.Series
        current filter id
    hist_mag : pd.Series
        magnitude estimation history over 30 days
    hist_diffmag : pd.Series
        upper limit estimation history
    h_jd : pd.Series
        julian date history of the measurements (upper limit of real detection)
    h_fid : pd.Series
        filter id history

    Returns
    -------
    abs_rate : double
        absolute rate, difference between the current magnitude estimation (real detection)
        and the last measurement in the history corresponding to the current filter id.
        The last measurement can be an upper limit or a real detection (S/N above 5)
    norm_rate : double
        normalized rate, absolute rate divided by the julian date of both measurements.
    first_variation_time : double
        first julian date of a real detection contains in the history.
    diff_start_hist : double
        difference between the first_variation_time and jdstarthist
    from_upper : boolean
        if the last measurement was an upper limit, return true (1.0) otherwise return false (0.0)

    Examples
    --------
    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 1,
    ... pd.Series([15, 16, 17, 18, 18, 19, 14, 12, 19, 19.5]),
    ... pd.Series([19, 19.8, 20, 18.2, 13.4, 15.4, 19.6, 17.5, 12.9, 15.8]),
    ... pd.Series([8, 9, 10, 11, 12, 14, 15, 19, 20, 24]),
    ... pd.Series([1, 1, 2, 2, 1, 2, 1, 2, 2, 1])
    ... ), decimals=3)
    array([ 5.5  ,  0.611,  8.   ,  3.   ,  0.   ])

    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 2,
    ... pd.Series([15, 16, 17, 18, 18, 19, 14, 12, 19, 19.5]),
    ... pd.Series([19, 19.8, 20, 18.2, 13.4, 15.4, 19.6, 17.5, 12.9, 15.8]),
    ... pd.Series([8, 9, 10, 11, 12, 14, 15, 19, 20, 24]),
    ... pd.Series([1, 1, 2, 2, 1, 2, 1, 2, 2, 2])
    ... ), decimals=3)
    array([ 0.5  ,  0.125,  8.   ,  3.   ,  0.   ])

    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 2,
    ... pd.Series([15, 16, 17, 18, 18, 19, 14, 12, np.nan, 19.5]),
    ... pd.Series([19, 19.8, 20, 18.2, 13.4, 15.4, 19.6, 17.5, 12.9, 15.8]),
    ... pd.Series([8, 9, 10, 11, 12, 14, 15, 19, 20, 24]),
    ... pd.Series([1, 1, 2, 2, 1, 2, 1, 2, 2, 2])
    ... ), decimals=3)
    array([ 6.6 ,  1.65,  8.  ,  3.  ,  1.  ])

    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 1,
    ... pd.Series([15, 16, 17, 18, 18, 19, np.nan, 12, np.nan, 19.5]),
    ... pd.Series([19, 19.8, 20, 18.2, 13.4, 15.4, 19.6, 17.5, 12.9, 15.8]),
    ... pd.Series([8, 9, 10, 11, 12, 14, 15, 19, 20, 24]),
    ... pd.Series([1, 1, 2, 2, 1, 2, 1, 2, 2, 2])
    ... ), decimals=3)
    array([-0.1  , -0.011,  8.   ,  3.   ,  1.   ])

    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 1,
    ... pd.Series([np.nan, 19.5]),
    ... pd.Series([12.9, 15.8]),
    ... pd.Series([20, 24]),
    ... pd.Series([2, 2])
    ... ), decimals=3)
    array([ 19.5,   nan,  24. ,   0. ,   0. ])

    >>> np.around(sub_compute_rate(
    ... 19.5,
    ... 5,
    ... 24,
    ... 2,
    ... pd.Series([np.nan, 19.5]),
    ... pd.Series([12.9, 15.8]),
    ... pd.Series([20, 24]),
    ... pd.Series([2, 2])
    ... ), decimals=3)
    array([  6.6 ,   1.65,  24.  ,  19.  ,   1.  ])
    """
    # 1 = g band, 2 = r band

    x = np.asarray(h_fid == curr_fid).nonzero()

    if len(x) > 0 and len(x[0]) > 1:
        idx_last_measurement_current_band = x[0][-1]
        if idx_last_measurement_current_band == len(hist_mag) - 1:
            idx_last_measurement_current_band = x[0][-2]
    else:
        return mag, np.nan, curr_jd, 0, False

    last_mag_curr_band = hist_mag[idx_last_measurement_current_band]
    last_diffmag_curr_band = hist_diffmag[idx_last_measurement_current_band]
    last_jd_curr_band = h_jd[idx_last_measurement_current_band]

    idx_non_nan_measurement = np.asarray(~np.isnan(hist_mag)).nonzero()
    idx_first_measurement = idx_non_nan_measurement[0][0]
    first_variation_time = h_jd[idx_first_measurement]
    diff_start_hist = first_variation_time - jdstarthist

    if np.isnan(last_mag_curr_band):
        abs_rate = mag - last_diffmag_curr_band
        norm_rate = abs_rate / (curr_jd - last_jd_curr_band)
        from_upper = True
        return abs_rate, norm_rate, first_variation_time, diff_start_hist, from_upper
    else:
        abs_rate = mag - last_mag_curr_band
        norm_rate = abs_rate / (curr_jd - last_jd_curr_band)
        from_upper = False
        return abs_rate, norm_rate, first_variation_time, diff_start_hist, from_upper


@pandas_udf(ArrayType(DoubleType()))
def compute_rate(
    magpsf, jdstarthist, jd, fid, hist_magpf, hist_difmaglim, hist_jd, hist_fid
):
    """
    see sub_compute_rate function documentation

    Examples
    --------
    >>> df_spark = spark.read.format("parquet").load(
    ... data_fid_1
    ... )

    >>> df_spark = concat_col(df_spark, "magpsf")
    >>> df_spark = concat_col(df_spark, "diffmaglim")
    >>> df_spark = concat_col(df_spark, "jd")
    >>> df_spark = concat_col(df_spark, "fid")

    >>> df_spark = df_spark.withColumn(
    ... "c_rate",
    ... compute_rate(
    ...     df_spark["candidate.magpsf"],
    ...     df_spark["candidate.jdstarthist"],
    ...     df_spark["candidate.jd"],
    ...     df_spark["candidate.fid"],
    ...     df_spark["cmagpsf"],
    ...     df_spark["cdiffmaglim"],
    ...     df_spark["cjd"],
    ...     df_spark["cfid"],
    ...     ),
    ... )

    >>> df_spark = format_rate_results(df_spark, "c_rate")
    >>> df_spark.select(
    ... "objectId",
    ... "delta_mag",
    ... "rate",
    ... "from_upper",
    ... "start_vartime",
    ... "diff_vartime"
    ... ).show()
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |    objectId|           delta_mag|                rate|from_upper|  start_vartime|      diff_vartime|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |ZTF22aayqeuc|  1.7709178924560547|  0.5873353005026273|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqeez|  0.8230018615722656|  0.2728675002256756|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqebm|    -0.9632568359375|-0.31936924710647197|       1.0|2459795.8062153|               0.0|
    |ZTF18abvorsh| 0.21074295043945312| 0.06987214095328952|       1.0|2459782.8572222|1522.8982870001346|
    |ZTF19aayjkvl|  0.6514949798583984| 0.21600413664183235|       1.0|2459770.8325694| 1168.845104099717|
    |ZTF22aayqeiq|-0.04056739807128906|-0.01345018160861...|       1.0|2459795.8062153|               0.0|
    |ZTF18abtntna|  0.2892932891845703| 0.09591562344835082|       1.0| 2459775.918044|1515.9591088001616|
    |ZTF22aayqegg|   0.356536865234375| 0.11821033183198935|       1.0|2459795.8062153|               0.0|
    |ZTF19acgekhi| -1.1580257415771484|-0.38394516957407226|       1.0|2459789.8571065|1013.2489351998083|
    |ZTF18abvorvl| -0.4214363098144531|-0.13972783991486226|       0.0|2459782.7935648| 1519.855416700244|
    |ZTF18abutrfi| -1.7925453186035156| -0.5943210859743961|       1.0|2459770.8325694|1510.8736342000775|
    |ZTF19addeyme|  1.0170612335205078|  0.3372081757348668|       1.0|2459766.8565394| 1158.937986199744|
    |ZTF18abobkzy| 0.10790824890136719| 0.10880742255460157|       0.0|2459765.8352894|1505.8763542002998|
    |ZTF18abobkum|  1.1519756317138672| 0.38193924658462675|       0.0|2459766.8565394|1506.8976042000577|
    |ZTF22aayqeei|-0.02435684204101...|-0.00807554747011...|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqess| -0.7202816009521484|-0.23881044392158104|       1.0|2459795.8062153|               0.0|
    |ZTF18acsvony|  0.9074382781982422|  0.3008625206606577|       1.0|2459766.8565394|1310.2850810997188|
    |ZTF18abmrejh| -0.5489215850830078| -0.2720457607210523|       0.0|2459765.8794213| 1503.920046299696|
    |ZTF20acnshgu| 0.22086524963378906|  0.1094609075071864|       0.0|2459765.8794213| 1482.938784699887|
    |ZTF18abmagok| -0.2481231689453125|-0.12296994339918015|       0.0|2459768.8373148|1521.8825809997506|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    only showing top 20 rows
    <BLANKLINE>

    >>> df_spark = spark.read.format("parquet").load(
    ... data_fid_2
    ... )

    >>> df_spark = concat_col(df_spark, "magpsf")
    >>> df_spark = concat_col(df_spark, "diffmaglim")
    >>> df_spark = concat_col(df_spark, "jd")
    >>> df_spark = concat_col(df_spark, "fid")

    >>> df_spark = df_spark.withColumn(
    ... "c_rate",
    ... compute_rate(
    ...     df_spark["candidate.magpsf"],
    ...     df_spark["candidate.jdstarthist"],
    ...     df_spark["candidate.jd"],
    ...     df_spark["candidate.fid"],
    ...     df_spark["cmagpsf"],
    ...     df_spark["cdiffmaglim"],
    ...     df_spark["cjd"],
    ...     df_spark["cfid"],
    ...     ),
    ... )

    >>> df_spark = format_rate_results(df_spark, "c_rate")
    >>> df_spark.select(
    ... "objectId",
    ... "delta_mag",
    ... "rate",
    ... "from_upper",
    ... "start_vartime",
    ... "diff_vartime"
    ... ).show()
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |    objectId|           delta_mag|                rate|from_upper|  start_vartime|      diff_vartime|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |ZTF22aatwlts| -0.5056400299072266|-0.04216281783616...|       1.0|2459777.6860069|               0.0|
    |ZTF18acwzatv|  1.3946971893310547|  0.3507262916342683|       1.0|2459747.7789815|1527.9326621000655|
    |ZTF18acuehhl|0.025455474853515625|0.008617208969552792|       0.0|2459747.6894792|1527.8736227001064|
    |ZTF20aakcvst|-0.12790489196777344|-0.04329847267265403|       1.0|2459747.6894792| 870.6672917003743|
    |ZTF22aatwmvj|-0.07059669494628906|-0.02389845313877...|       1.0|2459777.6864815|               0.0|
    |ZTF22aatwmks| -0.6354446411132812|-0.21511125966284503|       1.0|2459777.6864815|               0.0|
    |ZTF18acufbjc| -1.5980262756347656| -0.5409652122077429|       1.0|2459747.6894792|1527.8736227001064|
    |ZTF19abguwul| 0.17145156860351562| 0.09001066551612298|       0.0|2459749.7692014|1262.7115278001875|
    |ZTF22aatwnab|  0.5881366729736328|  0.3087669233940713|       1.0|2459777.6869676|               0.0|
    |ZTF20abfxzae|  1.7185096740722656|    0.87071320516342|       1.0|    2459747.715|1165.9422337999567|
    |ZTF20aabqhex|-0.07990264892578125|-0.04048408490035005|       1.0|    2459747.715| 896.7108911997639|
    |ZTF19aczkimj|  0.2899646759033203| 0.14691571199693776|       1.0|    2459747.715|1259.6964004999027|
    |ZTF22aatwmpl| 0.14156150817871094| 0.03566758552080276|       1.0|2459777.6879398|               0.0|
    |ZTF18adkhuxp|  -4.048986434936523|  -2.051490317230846|       1.0|2459749.7163657|1534.8256133999676|
    |ZTF20abgonhb| 0.37218475341796875|  0.1880149612952558|       1.0|2459749.7730903|1140.9793634000234|
    |ZTF22aatworv| -1.2060070037841797| -0.6092333392379711|       1.0|2459777.6885069|               0.0|
    |ZTF22aatwoek|  0.7335643768310547|  0.3705715418239658|       1.0|2459777.6885069|               0.0|
    |ZTF18adbafqc| 0.12570953369140625| 0.06350414114060082|       1.0| 2459747.794456|1501.0214466997422|
    |ZTF20abanmzg|  1.1241226196289062|   0.567868159239987|       1.0|2459777.6885069| 789.8739582998678|
    |ZTF22aatwouc|  1.3432426452636719|  0.6785600743719058|       1.0|2459777.6885069|               0.0|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    only showing top 20 rows
    <BLANKLINE>
    """
    t = [
        sub_compute_rate(
            mag, jd_start_hist, curr_jd, curr_fid, hist_mag, hist_diffmag, h_jd, h_fid
        )
        for mag, jd_start_hist, curr_jd, curr_fid, hist_mag, hist_diffmag, h_jd, h_fid in zip(
            magpsf, jdstarthist, jd, fid, hist_magpf, hist_difmaglim, hist_jd, hist_fid
        )
    ]

    return pd.Series(t)


def format_rate_results(spark_df, rate_column):
    """
    Extract the column return by compute_rate and add the following columns
        abs_rate, norm_rate, from_upper, start_vartime, diff_vartime

    Parameters
    ----------
    spark_df : spark dataframe
        dataframe containing alerts informations
    rate_column : string
        the name of the column containing the product from the compute_rate function.

    Returns
    -------
    spark_df : spark dataframe
        like the original dataframe but with the additional columns from compute_rate function

    Examples
    --------
    >>> df_spark = spark.read.format("parquet").load(
    ... data_fid_1
    ... )

    >>> df_spark = concat_col(df_spark, "magpsf")
    >>> df_spark = concat_col(df_spark, "diffmaglim")
    >>> df_spark = concat_col(df_spark, "jd")
    >>> df_spark = concat_col(df_spark, "fid")

    >>> df_spark = df_spark.withColumn(
    ... "c_rate",
    ... compute_rate(
    ...     df_spark["candidate.magpsf"],
    ...     df_spark["candidate.jdstarthist"],
    ...     df_spark["candidate.jd"],
    ...     df_spark["candidate.fid"],
    ...     df_spark["cmagpsf"],
    ...     df_spark["cdiffmaglim"],
    ...     df_spark["cjd"],
    ...     df_spark["cfid"],
    ...     ),
    ... )

    >>> df_spark = format_rate_results(df_spark, "c_rate")
    >>> df_spark.select(
    ... "objectId",
    ... "delta_mag",
    ... "rate",
    ... "from_upper",
    ... "start_vartime",
    ... "diff_vartime"
    ... ).show()
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |    objectId|           delta_mag|                rate|from_upper|  start_vartime|      diff_vartime|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |ZTF22aayqeuc|  1.7709178924560547|  0.5873353005026273|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqeez|  0.8230018615722656|  0.2728675002256756|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqebm|    -0.9632568359375|-0.31936924710647197|       1.0|2459795.8062153|               0.0|
    |ZTF18abvorsh| 0.21074295043945312| 0.06987214095328952|       1.0|2459782.8572222|1522.8982870001346|
    |ZTF19aayjkvl|  0.6514949798583984| 0.21600413664183235|       1.0|2459770.8325694| 1168.845104099717|
    |ZTF22aayqeiq|-0.04056739807128906|-0.01345018160861...|       1.0|2459795.8062153|               0.0|
    |ZTF18abtntna|  0.2892932891845703| 0.09591562344835082|       1.0| 2459775.918044|1515.9591088001616|
    |ZTF22aayqegg|   0.356536865234375| 0.11821033183198935|       1.0|2459795.8062153|               0.0|
    |ZTF19acgekhi| -1.1580257415771484|-0.38394516957407226|       1.0|2459789.8571065|1013.2489351998083|
    |ZTF18abvorvl| -0.4214363098144531|-0.13972783991486226|       0.0|2459782.7935648| 1519.855416700244|
    |ZTF18abutrfi| -1.7925453186035156| -0.5943210859743961|       1.0|2459770.8325694|1510.8736342000775|
    |ZTF19addeyme|  1.0170612335205078|  0.3372081757348668|       1.0|2459766.8565394| 1158.937986199744|
    |ZTF18abobkzy| 0.10790824890136719| 0.10880742255460157|       0.0|2459765.8352894|1505.8763542002998|
    |ZTF18abobkum|  1.1519756317138672| 0.38193924658462675|       0.0|2459766.8565394|1506.8976042000577|
    |ZTF22aayqeei|-0.02435684204101...|-0.00807554747011...|       1.0|2459795.8062153|               0.0|
    |ZTF22aayqess| -0.7202816009521484|-0.23881044392158104|       1.0|2459795.8062153|               0.0|
    |ZTF18acsvony|  0.9074382781982422|  0.3008625206606577|       1.0|2459766.8565394|1310.2850810997188|
    |ZTF18abmrejh| -0.5489215850830078| -0.2720457607210523|       0.0|2459765.8794213| 1503.920046299696|
    |ZTF20acnshgu| 0.22086524963378906|  0.1094609075071864|       0.0|2459765.8794213| 1482.938784699887|
    |ZTF18abmagok| -0.2481231689453125|-0.12296994339918015|       0.0|2459768.8373148|1521.8825809997506|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    only showing top 20 rows
    <BLANKLINE>

    >>> df_spark = spark.read.format("parquet").load(
    ... data_fid_2
    ... )

    >>> df_spark = concat_col(df_spark, "magpsf")
    >>> df_spark = concat_col(df_spark, "diffmaglim")
    >>> df_spark = concat_col(df_spark, "jd")
    >>> df_spark = concat_col(df_spark, "fid")

    >>> df_spark = df_spark.withColumn(
    ... "c_rate",
    ... compute_rate(
    ...     df_spark["candidate.magpsf"],
    ...     df_spark["candidate.jdstarthist"],
    ...     df_spark["candidate.jd"],
    ...     df_spark["candidate.fid"],
    ...     df_spark["cmagpsf"],
    ...     df_spark["cdiffmaglim"],
    ...     df_spark["cjd"],
    ...     df_spark["cfid"],
    ...     ),
    ... )

    >>> df_spark = format_rate_results(df_spark, "c_rate")
    >>> df_spark.select(
    ... "objectId",
    ... "delta_mag",
    ... "rate",
    ... "from_upper",
    ... "start_vartime",
    ... "diff_vartime"
    ... ).show()
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |    objectId|           delta_mag|                rate|from_upper|  start_vartime|      diff_vartime|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    |ZTF22aatwlts| -0.5056400299072266|-0.04216281783616...|       1.0|2459777.6860069|               0.0|
    |ZTF18acwzatv|  1.3946971893310547|  0.3507262916342683|       1.0|2459747.7789815|1527.9326621000655|
    |ZTF18acuehhl|0.025455474853515625|0.008617208969552792|       0.0|2459747.6894792|1527.8736227001064|
    |ZTF20aakcvst|-0.12790489196777344|-0.04329847267265403|       1.0|2459747.6894792| 870.6672917003743|
    |ZTF22aatwmvj|-0.07059669494628906|-0.02389845313877...|       1.0|2459777.6864815|               0.0|
    |ZTF22aatwmks| -0.6354446411132812|-0.21511125966284503|       1.0|2459777.6864815|               0.0|
    |ZTF18acufbjc| -1.5980262756347656| -0.5409652122077429|       1.0|2459747.6894792|1527.8736227001064|
    |ZTF19abguwul| 0.17145156860351562| 0.09001066551612298|       0.0|2459749.7692014|1262.7115278001875|
    |ZTF22aatwnab|  0.5881366729736328|  0.3087669233940713|       1.0|2459777.6869676|               0.0|
    |ZTF20abfxzae|  1.7185096740722656|    0.87071320516342|       1.0|    2459747.715|1165.9422337999567|
    |ZTF20aabqhex|-0.07990264892578125|-0.04048408490035005|       1.0|    2459747.715| 896.7108911997639|
    |ZTF19aczkimj|  0.2899646759033203| 0.14691571199693776|       1.0|    2459747.715|1259.6964004999027|
    |ZTF22aatwmpl| 0.14156150817871094| 0.03566758552080276|       1.0|2459777.6879398|               0.0|
    |ZTF18adkhuxp|  -4.048986434936523|  -2.051490317230846|       1.0|2459749.7163657|1534.8256133999676|
    |ZTF20abgonhb| 0.37218475341796875|  0.1880149612952558|       1.0|2459749.7730903|1140.9793634000234|
    |ZTF22aatworv| -1.2060070037841797| -0.6092333392379711|       1.0|2459777.6885069|               0.0|
    |ZTF22aatwoek|  0.7335643768310547|  0.3705715418239658|       1.0|2459777.6885069|               0.0|
    |ZTF18adbafqc| 0.12570953369140625| 0.06350414114060082|       1.0| 2459747.794456|1501.0214466997422|
    |ZTF20abanmzg|  1.1241226196289062|   0.567868159239987|       1.0|2459777.6885069| 789.8739582998678|
    |ZTF22aatwouc|  1.3432426452636719|  0.6785600743719058|       1.0|2459777.6885069|               0.0|
    +------------+--------------------+--------------------+----------+---------------+------------------+
    only showing top 20 rows
    <BLANKLINE>
    """
    return (
        spark_df.withColumn("delta_mag", F.col(rate_column).getItem(0))
        .withColumn("rate", F.col(rate_column).getItem(1))
        .withColumn("from_upper", F.col(rate_column).getItem(4))
        .withColumn("start_vartime", F.col(rate_column).getItem(2))
        .withColumn("diff_vartime", F.col(rate_column).getItem(3))
        .drop(rate_column)
    )


def join_post_process(df_grb, with_rate=True, from_hbase=False):

    if with_rate:

        df_grb = concat_col(df_grb, "magpsf")
        df_grb = concat_col(df_grb, "diffmaglim")
        df_grb = concat_col(df_grb, "jd")
        df_grb = concat_col(df_grb, "fid")

        df_grb = df_grb.withColumn(
            "c_rate",
            compute_rate(
                df_grb["{}magpsf".format("" if from_hbase else "candidate.")],
                df_grb["{}jdstarthist".format("" if from_hbase else "candidate.")],
                df_grb["{}jd".format("" if from_hbase else "candidate.")],
                df_grb["{}fid".format("" if from_hbase else "candidate.")],
                df_grb["cmagpsf"],
                df_grb["cdiffmaglim"],
                df_grb["cjd"],
                df_grb["cfid"],
            ),
        )

        df_grb = format_rate_results(df_grb, "c_rate")

    df_grb = df_grb.withColumn(
        "fink_class",
        extract_fink_classification(
            df_grb["cdsxmatch"],
            df_grb["roid"],
            df_grb["mulens"],
            df_grb["snn_snia_vs_nonia"],
            df_grb["snn_sn_vs_all"],
            df_grb["rf_snia_vs_nonia"],
            df_grb["{}ndethist".format("" if from_hbase else "candidate.")],
            df_grb["{}drb".format("" if from_hbase else "candidate.")],
            df_grb["{}classtar".format("" if from_hbase else "candidate.")],
            df_grb["{}jd".format("" if from_hbase else "candidate.")],
            df_grb["{}jdstarthist".format("" if from_hbase else "candidate.")],
            df_grb["rf_kn_vs_nonkn"],
            df_grb["tracklet"],
        ),
    )

    # refine the association and compute the serendipitous probability
    df_grb = df_grb.withColumn(
        "grb_proba",
        grb_assoc(
            df_grb["ztf_ra"],
            df_grb["ztf_dec"],
            df_grb["{}".format("start_vartime" if with_rate else "jdstarthist")],
            df_grb["platform"],
            df_grb["triggerTimeUTC"],
            df_grb["grb_ra"],
            df_grb["grb_dec"],
            df_grb["err_arcmin"],
        ),
    )

    column_to_return = [
        "objectId",
        "candid",
        "ztf_ra",
        "ztf_dec",
        "{}fid".format("" if from_hbase else "candidate."),
        "{}jdstarthist".format("" if from_hbase else "candidate."),
        "{}rb".format("" if from_hbase else "candidate."),
        "{}jd".format("" if from_hbase else "candidate."),
        "instrument_or_event",
        "platform",
        "triggerId",
        "grb_ra",
        "grb_dec",
        col("err_arcmin").alias("grb_loc_error"),
        "triggerTimeUTC",
        "grb_proba",
        "fink_class",
    ]

    if with_rate:

        column_to_return += [
            "delta_mag",
            "rate",
            "from_upper",
            "start_vartime",
            "diff_vartime",
        ]

    # select a subset of columns before the writing
    df_grb = df_grb.select(column_to_return).filter("grb_proba != -1.0")

    return df_grb


if __name__ == "__main__":  # pragma: no cover
    import sys  # noqa: F401
    import doctest  # noqa: F401
    from pandas.testing import assert_frame_equal  # noqa: F401
    import pandas as pd  # noqa: F401
    import shutil  # noqa: F401
    from fink_grb.init import get_config, init_logging  # noqa: F401
    from fink_utils.spark.utils import concat_col  # noqa: F401

    from fink_utils.test.tester import spark_unit_tests_science

    globs = globals()

    path_data_fid_1 = "fink_grb/test/test_data/ztf_alert_samples_fid_1.parquet"
    path_data_fid_2 = "fink_grb/test/test_data/ztf_alert_samples_fid_2.parquet"
    globs["data_fid_1"] = path_data_fid_1
    globs["data_fid_2"] = path_data_fid_2

    # Run the test suite
    spark_unit_tests_science(globs)

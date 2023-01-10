import numpy as np
import pandas as pd
from math import pi
from scipy.stats import poisson

from pyspark.sql.functions import pandas_udf, col  # noqa: F401
from pyspark.sql.types import DoubleType

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time


def p_ser_grb_vect(
    error_radius: np.ndarray, size_time_window: np.ndarray, r_grb: np.ndarray
) -> list:
    """
    Created on Mon Oct  4 10:34:09 2021

    @author: Damien Turpin : damien.turpin@cea.fr

    function that gives the chance probability of having a positive spatial and
    temporal match between a GRB and a ZTF transient candidate

    Parameters
    ----------
    error_radius : array
        error radius of the GRB localization region in degree
    size_time_window: array
        size of the searching time window in year
    r_grb: array
        GRB detection rate for a set of satellites in events/year

    Returns
    -------
    p_ser : list
        Serendipituous probabilities for a GRB/ZTF candidate association.
        The first items correspond to the association probability with a GRB in general, the second correspond
        to the association with a long GRB and finally, the last items correspond to the associations with a
        short GRB.

    Examples
    --------

    # alerts from the object ZTF21aagwbjr
    >>> ztf_alerts = pd.read_parquet("fink_grb/test/test_data/ztf_alerts_sample.parquet")

    # gcn contains the notice 634112970 (GRB210204270)
    >>> grb_alerts = pd.read_parquet("fink_grb/test/test_data/grb_samples.parquet")

    >>> proba = grb_alerts.apply(
    ... lambda x: p_ser_grb_vect(
    ...     x["gm_error"],
    ...     (ztf_alerts["i:jdstarthist"].values - Time(x["Trig Time"].to_datetime64(), format="datetime64").jd) / 365.25,
    ...     250)[0][0],
    ... axis=1
    ... )

    >>> (1 - proba.values[1]) > special.erf(5 / sqrt(2))
    True
    """

    # omega = 2*pi*(1-cos(radians(error_radius))) # solid angle in steradians
    grb_loc_area = pi * np.power(error_radius, 2)  # in square degrees
    allsky_area = 4 * pi * (180 / pi) ** 2  # in square degrees
    ztf_coverage_rate = 3750  # sky coverage rate of ZTF in square degrees per hour
    limit_survey_time = 4  # duration (in hour) during which ZTF will cover individual parts of the sky in a night

    # short and long GRB detection rate
    r_sgrb = np.divide(r_grb, 3)
    r_lgrb = r_grb - r_sgrb

    # Poisson probability of detecting a GRB during a searching time window
    p_grb_detect_ser = 1 - poisson.cdf(1, r_grb * size_time_window)
    p_lgrb_detect_ser = 1 - poisson.cdf(1, r_lgrb * size_time_window)
    p_sgrb_detect_ser = 1 - poisson.cdf(1, r_sgrb * size_time_window)

    # we limit the fraction of the sky ZTF is able to cover to 4 hours of continuous survey
    # we consider that every day (during several days only) ZTF will cover the same part of
    # the sky with individual shots (so no revisit) during 4 hours

    #     if size_time_window*365.25*24 <= limit_survey_time:
    #         ztf_sky_frac_area = (ztf_coverage_rate*size_time_window*365.25*24)
    #     else:
    #         ztf_sky_frac_area = ztf_coverage_rate*limit_survey_time

    ztf_sky_frac_area = np.where(
        size_time_window * 365.25 * 24 <= limit_survey_time,
        (ztf_coverage_rate * size_time_window * 365.25 * 24),
        ztf_coverage_rate * limit_survey_time,
    )

    # probability of finding a GRB within the region area paved by ZTF during a given amount of time
    p_grb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_grb_detect_ser
    p_lgrb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_lgrb_detect_ser
    p_sgrb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_sgrb_detect_ser

    # probability of finding a ZTF transient candidate inside the GRB error box
    # knowing the GRB is in the region area paved by ZTF during a given amount of time

    p_ser_grb = p_grb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_ser_lgrb = p_lgrb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_ser_sgrb = p_sgrb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_sers = [p_ser_grb, p_ser_lgrb, p_ser_sgrb]

    return p_sers


@pandas_udf(DoubleType())
def grb_assoc(
    ztf_ra: pd.Series,
    ztf_dec: pd.Series,
    jdstarthist: pd.Series,
    platform: pd.Series,
    trigger_time: pd.Series,
    grb_ra: pd.Series,
    grb_dec: pd.Series,
    grb_error: pd.Series,
) -> pd.Series:
    """
    Find the ztf alerts falling in the error box of the notices and emits after the trigger time.
    Then, Compute an association serendipitous probability for each of them and return it.

    Parameters
    ----------
    ztf_ra : double spark column
        right ascension coordinates of the ztf alerts
    ztf_dec : double spark column
        declination coordinates of the ztf alerts
    jdstarthist : double spark column
        Earliest Julian date of epoch corresponding to ndethist [days]
        ndethist : Number of spatially-coincident detections falling within 1.5 arcsec
            going back to beginning of survey;
            only detections that fell on the same field and readout-channel ID
            where the input candidate was observed are counted.
            All raw detections down to a photometric S/N of ~ 3 are included.
    platform : string spark column
        voevent emitting platform
    trigger_time : double spark column
        grb trigger time (UTC)
    grb_ra : double spark column
        grb right ascension
    grb_dec : double spark column
        grb declination
    grb_error : double spark column
        grb error radius (in arcminute)

    Returns
    -------
    grb_proba : pandas Series
        the serendipitous probability for each ztf alerts.

    Examples
    --------

    >>> sparkDF = spark.read.format('parquet').load(join_data)

    >>> df_grb = sparkDF.withColumn(
    ... "grb_proba",
    ... grb_assoc(
    ...    sparkDF.candidate.ra,
    ...     sparkDF.candidate.dec,
    ...     sparkDF.candidate.jdstarthist,
    ...     sparkDF.platform,
    ...     sparkDF.timeUTC,
    ...     sparkDF.ra,
    ...     sparkDF.dec,
    ...     sparkDF.err
    ...  ),
    ... )

    >>> df_grb = df_grb.select([
    ... "objectId",
    ... "candid",
    ... col("candidate.ra").alias("ztf_ra"),
    ... col("candidate.dec").alias("ztf_dec"),
    ... "candidate.jd",
    ... "platform",
    ... "instrument",
    ... "trigger_id",
    ... col("ra").alias("grb_ra"),
    ... col("dec").alias("grb_dec"),
    ... col("err_arcmin").alias("grb_loc_error"),
    ... "timeUTC",
    ... "grb_proba"
    ... ])

    >>> grb_prob = df_grb.toPandas()
    >>> grb_test = pd.read_parquet("fink_grb/test/test_data/grb_prob_test.parquet")
    >>> assert_frame_equal(grb_prob, grb_test)
    """
    grb_proba = np.ones_like(ztf_ra.values, dtype=float) * -1.0
    platform = platform.values

    # array of events detection rates in events/years
    # depending of the instruments
    condition = [
        np.equal(platform, "Fermi"),
        np.equal(platform, "SWIFT"),
        np.equal(platform, "INTEGRAL"),
        np.equal(platform, "ICECUBE"),
    ]
    choice_grb_rate = [250, 100, 60, 8]
    grb_det_rate = np.select(condition, choice_grb_rate)

    # array of error box
    grb_error = grb_error.values

    trigger_time = Time(
        pd.to_datetime(trigger_time.values, utc=True), format="datetime"
    ).jd

    # alerts emits after the grb
    delay = jdstarthist - trigger_time
    time_condition = delay > 0

    ztf_coords = SkyCoord(ztf_ra, ztf_dec, unit=u.degree)
    grb_coord = SkyCoord(grb_ra, grb_dec, unit=u.degree)

    # alerts falling within the grb_error_box
    spatial_condition = (
        ztf_coords.separation(grb_coord).arcminute < 1.5 * grb_error
    )  # 63.5 * grb_error

    # convert the delay in year
    delay_year = delay[time_condition & spatial_condition] / 365.25

    # compute serendipitous probability
    p_ser = p_ser_grb_vect(
        grb_error[time_condition & spatial_condition] / 60,
        delay_year.values,
        grb_det_rate[time_condition & spatial_condition],
    )

    grb_proba[time_condition & spatial_condition] = p_ser[0]

    return pd.Series(grb_proba)


if __name__ == "__main__":  # pragma: no cover
    import pandas as pd  # noqa: F401
    from math import sqrt  # noqa: F401
    from scipy import special  # noqa: F401
    from fink_utils.test.tester import spark_unit_tests_science
    from pandas.testing import assert_frame_equal  # noqa: F401

    globs = globals()

    join_data = "fink_grb/test/test_data/join_raw_datatest.parquet"

    # Run the test suite
    spark_unit_tests_science(globs)

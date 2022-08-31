import signal
import pyarrow as pa
import pyarrow.parquet as pq

from gcn_kafka import Consumer
from fink_grb.online.instruments import LISTEN_PACKS, INSTR_SUBSCRIBES

import io
import logging

import fink_grb.online.gcn_reader as gr
from fink_grb.init import get_config

from fink_grb import __name__

def signal_handler(signal, frame):
    logging.warn("exit the gcn streaming !")
    exit(0)


def init_logging():

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n\t message: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


def start_gcn_stream(arguments):

    config = get_config(arguments)
    logger = init_logging()

    # Connect as a consumer.
    # Warning: don't share the client secret with others.
    consumer = Consumer(
        client_id=config["CLIENT"]["id"], client_secret=config["CLIENT"]["secret"]
    )

    # Subscribe to topics and receive alerts
    consumer.subscribe(INSTR_SUBSCRIBES)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        message = consumer.consume(timeout=2)

        if len(message) != 0:
            for gcn in message:
                logger.info("A new voevent is coming")
                value = gcn.value()
                
                decode = io.BytesIO(value).read().decode("UTF-8")

                try:
                    voevent = gr.load_voevent(io.StringIO(decode))
                except Exception as e:
                    logger.error("Error while reading the following voevent: \n\t {}\n\n\tcause: {}".format(decode, e))
                    print()
                    continue

                if gr.is_observation(voevent) and gr.is_listened_packets_types(voevent, LISTEN_PACKS):
                    
                    logging.info("the voevent is a new obervation.")

                    df = gr.voevent_to_df(voevent)

                    df['year'] = df['timeUTC'].dt.strftime('%Y')
                    df['month'] = df['timeUTC'].dt.strftime('%m')
                    df['day'] = df['timeUTC'].dt.strftime('%d')

                    table = pa.Table.from_pandas(df)

                    pq.write_to_dataset(
                        table,
                        root_path=config["PATH"]["gcn_path_storage"],
                        partition_cols=['year', 'month', 'day'],
                        basename_template="{}_{}".format(str(df["trigger_id"].values[0]), "{i}"),
                        existing_data_behavior="overwrite_or_ignore"
                    )

                    logging.info("writing of the new voevent successfull at the location {}".format(config["PATH"]["gcn_path_storage"]))

#!/usr/bin/env python

from datetime import datetime
import logging
import os

import requests

from cassandra import ConsistencyLevel
from cassandra.io.libevreactor import LibevConnection
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

# setup our logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# the number of allowed pending futures
# arbitrarily small for effect
MAX_FUTURE_QUEUE = 5000

# if running at a conference, only use coin_ids
# if running outside of a conference, set to false to import all cryptocurrencies
PRESENTATION = True

# for Linux time calculations
EPOCH = datetime.utcfromtimestamp(0).date()

# define which coins and years to track
coins = []
coin_ids = ['aeon', 'agoras-tokens', 'bitbay', 'bitcny', 'bitcoin',
            'bitcoindark', 'bitcrystals', 'bitsend', 'blackcoin', 'blocknet',
            'burst', 'clams', 'counterparty', 'crown', 'dash', 'decred',
            'diamond', 'digibyte', 'digitalnote', 'dimecoin', 'dogecoin',
            'einsteinium', 'emercoin', 'energycoin', 'ethereum', 'expanse',
            'factom', 'faircoin', 'feathercoin', 'florincoin', 'foldingcoin',
            'gambit', 'gamecredits', 'gridcoin', 'groestlcoin', 'gulden',
            'hempcoin', 'iocoin', 'jinn', 'litecoin', 'maidsafecoin',
            'mintcoin', 'monacoin', 'monero', 'monetaryunit', 'mooncoin',
            'namecoin', 'nav-coin', 'nem', 'neoscoin', 'newyorkcoin', 'nimiq',
            'nushares', 'nxt', 'okcash', 'omni', 'paccoin', 'pandacoin-pnd',
            'peercoin', 'potcoin', 'radium', 'reddcoin', 'ripple', 'rubycoin',
            'salus', 'shift', 'sibcoin', 'solarcoin', 'stealthcoin', 'stellar',
            'supernet-unity', 'synereo', 'verge', 'vericoin', 'vertcoin',
            'viacoin', 'voxels', 'whitecoin']
# back up presentation list, for very slow internet connection
coin_ids = ['bitcoin', 'ethereum', 'factom', 'litecoin', 'maidsafecoin',
            'monero', 'nem', 'stellar', 'verge']
for coin in coin_ids:
    coins.append({coin: [2016, 2017, 2018]})


def get_session():
    '''
    Connect onto a Cassandra cluster with the driver.
    :return: A Cassandra session object for cluster interactions.
    '''

    # grab the cluster information using Docker-provided enviornmental variables
    CASSANDRA_HOST = os.environ['CASSANDRA_HOST']
    CASSANDRA_DC = os.environ['CASSANDRA_DC']

    # create a cluster object that will only connect to a single data center
    cluster = Cluster([CASSANDRA_HOST],
                      load_balancing_policy=DCAwareRoundRobinPolicy(
                          local_dc=CASSANDRA_DC), )

    # use the faster event loop provider
    cluster.connection_class = LibevConnection

    # create the Cassandra session for cluster interaction
    session = cluster.connect()
    return session


def prepare_statements(session):
    '''
    Prepare text-based statements to convert long texts into short ints.
    :param session: Cassandra session.
    :return: Two prepared statements for inserting data into Cassandra.
    '''
    prepared_category_insertion = session.prepare('''
    INSERT INTO crypto.historical
      (coin, category)
    VALUES
      (?, ?)
    ''')
    prepared_category_insertion.consistency_level = ConsistencyLevel.QUORUM

    prepared_value_insertion = session.prepare('''
    INSERT INTO crypto.historical
      (coin, timestamp, value)
    VALUES
      (?, ?, ?)
    ''')
    prepared_value_insertion.consistency_level = ConsistencyLevel.QUORUM

    prepared_coin_deletion = session.prepare('''
    DELETE FROM crypto.historical
    WHERE coin = ?
    ''')
    prepared_coin_deletion.consistency_level = ConsistencyLevel.QUORUM

    return prepared_category_insertion, \
           prepared_value_insertion, \
           prepared_coin_deletion


def main():
    # create Cassandra session and prepared statements
    session = get_session()
    category_insertion, value_insertion, coin_deletion = prepare_statements(
        session)

    # hold all pending futures
    futures = []

    # comment out for better conference support
    if not PRESENTATION:
        del coins[:]
        logger.info('Grabbing a list of all cryptocurrencies...')
        response = requests.get('http://coinmarketcap.northpole.ro/coins.json')
        json_response = response.json()
        logger.info('Found %s cryptocurrencies...',
                    len(json_response['coins']))
        logger.info('Filtering for long-lived cryptocurrencies...')
        for coin in json_response['coins']:
            if not '2016' in coin['periods'] \
                    or not '2018' in coin['periods']:
                continue
            periods = coin['periods']
            periods.remove('14days')
            coins.append({coin['identifier']: periods})

    logger.info('Loading %s cryptocurrencies\' historical data...', len(coins))
    for i, coin in enumerate(coins):
        years = coin.values()[0]
        coin = coin.keys()[0]
        for year in years:
            # hit a web-based API to grab historical cryptocurrency prices
            logging.info('Grabbing %s historical data for: %s [%s/%s]...',
                         year, coin, i, len(coins))
            try:
                response = requests.get(
                    "http://coinmarketcap.northpole.ro/history.json?coin=%s&period=%s"
                    % (coin, year))
            except KeyboardInterrupt:
                exit(1)
            except:
                future = session.execute_async(coin_deletion, (coin,))
                futures.append(future)
                logger.exception('Failed to get API response')
                break

            try:
                json_response = response.json()
            except:
                future = session.execute_async(coin_deletion, (coin,))
                futures.append(future)
                logger.error(response)
                break

            if 'error' in json_response:
                continue

            # save STATIC category information once to reduce over-the-wire costs
            category = json_response['history'][0]['category']
            future = session.execute_async(category_insertion,
                                           (coin, category))
            futures.append(future)

            for history in json_response['history']:
                # datetime maths to strip hour:minute and convert back to Linux time
                timestamp = (
                        datetime.fromtimestamp(history['timestamp']).date()
                        - EPOCH).total_seconds()

                # save information from API response
                future = session.execute_async(value_insertion,
                                               (coin, timestamp,
                                                history['price']['usd']))
                futures.append(future)

                # ensure we never overload Cassandra with arbitrary throttling
                if len(futures) > MAX_FUTURE_QUEUE:
                    logging.info('Committing %s writes...', len(futures))
                    while futures:
                        future = futures.pop()
                        future.result()

    # ensure all in-flight writes are accepted before exiting
    logging.info('Flushing all %s pending writes...', len(futures))
    while futures:
        future = futures.pop()
        future.result()

    logging.info('All historical data saved into Cassandra!')


if __name__ == '__main__':
    main()

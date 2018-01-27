CREATE KEYSPACE IF NOT EXISTS crypto
WITH replication = {'class': 'NetworkTopologyStrategy', 'datacenter-1': 1 };

CREATE TABLE IF NOT EXISTS crypto.historical (
  coin text,
  timestamp int,
  value float,
  category text STATIC,
  PRIMARY KEY ((coin), timestamp)
);

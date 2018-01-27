# Data Day 2018

## Using Dockerized Cassandra and TensorFlow to Predict Future Blockchain Prices

Join Joaquin Casares of The Last Pickle in a code-heavy presentation of how he uses Docker Compose to start all of his new projects for his day job, clients, and side projects.

The presentation will come with a companion Github repository that contains a Docker Compose setup with Cassandra as well as a TensorFlow app to ingest and analyze blockchain technology price data.

To get the most out of the project, we recommend installing the following software before the meeting:

* Docker Engine (for Mac): https://docs.docker.com/docker-for-mac/install/
* Docker Engine (Ubuntu): https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/
* Docker Compose: https://docs.docker.com/compose/install/


### Commands

```bash
# build Docker images
docker-compose build

# start Cassandra
docker-compose up cassandra

# check on Cassandra
docker-compose exec cassandra nodetool -h cassandra status

# create Cassandra schema
docker-compose run load-schema

# load cryptocurrency data into Cassandra
docker-compose run load-data

# poke at Cassandra data
docker-compose run cqlsh

# use TensorFlow to analyze data
docker-compose run analyze

# view Machine Intelligence result
open result.png
```

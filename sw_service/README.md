# CL_for_ImageSegmentation_APP

![img](../assets/structure.png)

## docker compose
using docker compose to setup for `redis`, `minio` & `postgresql`

```shell
docker-compose up -d
```

## Minio Setup

create two bucket named `dataset` & `model`, change access policy to `public`.

## Celery startup

```shell
## start celery worker
celery -A celery_worker worker --loglevel=info --concurrency=1 --pool=solo
```

## Fastapi startup

```shell
## start fastapi app
python app.py
```

### API test

you can use the postman collection for api-testing. 

There are also some useful files for testing at [samples](./samples/)

- postman collection: `cl_app.postman_collection.json`
- dataset zip sample: `dataset.zip`


FROM postgres:14-alpine
ADD db/create_db.sql /docker-entrypoint-initdb.d

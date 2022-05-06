# syntax=docker/dockerfile:1
# author: Jiri Tumpach 

FROM redis

RUN redis-cli -a werystrongpassword


ENTRYPOINT ["redis-server"]




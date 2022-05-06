
import redis
import os
import unittest

print(os.getuid())


class Server:
    pass


class RedisServer:
    TRANSLATOR = {
        'push_jobs': 0,
        'get_job'  : 0,
        'push_results' : 1
    }

    def _get_host():
        return 'localhost'
    def _get_port():
        return 6379
    def _get_db():
        return RedisServer.TRANSLATOR[self.type_of_connection]
    def _get_password():
        return 'af75642febdbc9e79210d76f5214f7243f646852'

    def __init__(self, type_of_connection:str):
        assert type_of_connection in RedisServer.TRANSLATOR
        self.type_of_connection = type_of_connection

        self.connection = redis.Redis(
            host=self._get_host(),
            port=self._get_port(),
            db=self._get_db(),
            password=self._get_password()
        )

    def push_jobs(self, hyperparameters: dict):
        with self.connection.pipe() as p:
            p.sadd(


    def push_data(self, hyperparameters: dict, results: dict):
        with self.connection.pipe() as p:
            pass

    def get_data(hyperparameters):
        with self.connection.pipe() as p:
            pass

        



if __name__ == '__main__':
    class TestAddJobs(unittest.TestCase):
        def testAddOne(self):
            rs = RedisServer('push_jobs')
            for hp1 in range(10):
                for hp2 in ['first', 'third', 'second']:
                    rs.push_data({'hp1':hp1, 'hp2':hp2})




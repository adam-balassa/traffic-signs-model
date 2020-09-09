import zerorpc
from server.application import Application
from server.constants import PORT_NUMBER, LOCALHOST


server = zerorpc.Server(Application())
server.bind('{}:{}'.format(LOCALHOST, PORT_NUMBER))
print(f'Server started on port {PORT_NUMBER}')


def run():
    server.run()

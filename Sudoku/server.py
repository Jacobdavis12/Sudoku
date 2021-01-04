from http.server import BaseHTTPRequestHandler
import socketserver

class handler(BaseHTTPRequestHandler):
    header = '<html> <head> <title>Sudoku solver</title> <style>#header { font-size: 10vh; height: 10vh; margin: 0px; } #content { display: flex; flex-wrap: wrap; } #sudoku { height: 80vh; width: 80vh; display: grid; grid-template-columns: 26vh 26vh 26vh; grid-gap: 1vh; background-color: black; border:solid #cccccc 1vh; float:left; } .square { position: relative; height: 100%; width: 100%; display: grid; grid-template-columns: 8vh 8vh 8vh; grid-gap: 1vh; background-color: #c0c0c0; } .cell { font-size: 8vh; height: 8vh; text-align: center; background-color: white; border: 0px; } #image, .type { display: none; } .control { background-color: #004080; border-radius: 5px; border: 1px solid white; color: white; } #controls { flex-grow: 100; display: grid; grid-template-rows: 26vh 26vh 26vh; grid-gap: 1vh; white-space: nowrap; font-size: 6vh; padding: 1vh 0px 0px 1vh; } #controls > .control{ display: flex; height: 26vh; width: 100%; } @media screen and (orientation: portrait) { #header { font-size: 10vw; height: 10vw; } #sudoku { height: 80vw; width: 80vw; grid-template-columns: 26vw 26vw 26vw; grid-gap: 1vw; border-width: 1vw; } .square { grid-template-columns: 8vw 8vw 8vw; grid-gap: 1vw; } .cell { font-size: 8vw; height: 8vw; } #controls { padding-left: 0px; font-size: 6vw; height: 6vw; grid-template-rows: 26vw 26vw 26vw; grid-gap: 1vw; padding: 1vw 0px 0px 1vw; } #controls > .control{ height: 26vw; } } </style> </head> <body> <h1 id="header">Sudoku solver</h1> <div id="content"> <form id="sudoku" method="post"> <input name="type" class="type" value="sudoku"></input>'
    footer = '</form> <div id="controls"> <label for="image" class="control"> Upload Sudoku image <form id = "upload" method="post" enctype="multipart/form-data"> <input name="type" class="type" value="image"></input> <input id="image" type="file" name="filename" onchange="form.submit()"> </form> </label> <label class="control" for="photo"> Preferences <form id = "preference" method="post"> <input name="type" class="type" value="preference"></input> </form> </label> <label class="control" onclick="sudoku.submit()"> Solve Sudoku </label> </div> </div> </body> </html>'
    
    def sendHtml(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(data)#.encode("utf8"))
        
    def do_GET(self, params = {}):
        index = self.header
        for h in range(9):
            index += '<div id="s' + str(h) + '" class="square">\n'
            for w in range(9):
                if str(h) + str(w) in params:
                    index += '<input type="text" size="1" maxlength="1" name="' + str(h) + str(w) + '" value="' + params[str(h) + str(w)] + '" class="cell">\n'
                else:
                    index += '<input type="text" size="1" maxlength="1" name="' + str(h) + str(w) + '" value="" class="cell">\n'
            index += '</div>\n'

        index += self.footer
            
        self.sendHtml(index.encode("utf8"))
            
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        postData = self.rfile.read(content_length)
        #print([postData])
        if b'sudoku' in postData:
            self.do_GET(solveSudoku(postData))
        elif b'image' in postData:
            formatImage(postData)
            self.do_GET()
        elif b'preference' in postData:
            updatePref(postData)
        else:
            print('error')

def solveSudoku(rawData):
    rawData = str(rawData)
    data = {}
    for value in rawData[:-1].split('&')[1:]:
        values = value.split('=')
        data[values[0]] = values[1]
    print(data)
    return data

def formatImage(rawData):
    filename = rawData.split(b'filename="')[1].split(b'"')[0]
    data = rawData.split(b'\r\n\r\n')
    with open(filename, 'wb') as f:
        f.write(data[-1])

def updatePref(rawData):
    pass

addr = 'localhost'
port = 8080
serverAddress = (addr, port)

with socketserver.TCPServer(serverAddress, handler) as httpd:
    print(f'Starting httpd server on {addr}:{port}')
    httpd.serve_forever()

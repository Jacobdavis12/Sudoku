from http.server import BaseHTTPRequestHandler
import socketserver

class handler(BaseHTTPRequestHandler):
    html = ['<html><head><title>Sudoku solver</title><style>',
            '</style><script type="text/javascript">function openTab(tab) { document.getElementById("tab1").className = "buttons tab"; document.getElementById("tab2").className = "buttons tab"; document.getElementById("tab3").className = "buttons tab"; document.getElementById("content1").style.display = "none"; document.getElementById("content2").style.display = "none"; document.getElementById("content3").style.display = "none"; document.getElementById("tab" + String(tab)).className += " tabOpen"; document.getElementById("content" + String(tab)).style.display = "block"; }; function updatePrefs() { for (var i = 0; i < 3; i++) { if (document.getElementsByClassName("taskInput")[i].checked) { for (var j = 0; j < 3; j++) { document.getElementsByClassName("taskHiddenInput")[j].setAttribute("value", document.getElementsByClassName("taskInput")[i].getAttribute("value")); }; }; }; }; function sumbitSudoku() { updatePrefs(); sudoku.submit(); }; function submitVersion() { updatePrefs(); for (var i = 0; i < 3; i++) { if (document.getElementsByClassName("versionInput")[i].checked) { for (var j = 0; j < 3; j++) { document.getElementsByClassName("versionHiddenInput")[j].setAttribute("value", document.getElementsByClassName("versionInput")[i].getAttribute("value")); }; }; }; version.submit(); }; function submitUpload() { updatePrefs(); upload.submit(); };</script> </head><body><h1 id="header">Sudoku solver</h1><div id="content"><form id="sudoku" method="post"> <input name="type" class="hiddenInputs" value="sudoku"></input> <input name="version" class="hiddenInputs versionHiddenInput" value="',
            '"></input> <input name="task" class="hiddenInputs taskHiddenInput"></input>',
            '</form><div id="controls"> <label for="image" class="control buttons"> Upload Sudoku image<form id = "upload" method="post" enctype="multipart/form-data"> <input name="type" class="hiddenInputs" value="image"></input> <input name="version" class="hiddenInputs versionHiddenInput" value="', 
            '"></input> <input name="task" class="hiddenInputs taskHiddenInput"></input> <input id="image" type="file" name="filename" onchange="submitUpload()"></form> </label> <label id="prefernceControl" class="control" for="photo"><div id="tabs"><div id="tab0"> Preferences</div><div id="tab1" class="buttons tab tabOpen" onclick="openTab(1)"> Menu</div><div id="tab2" class="buttons tab" onclick="openTab(2)"> Version</div><div id="tab3" class="buttons tab" onclick="openTab(3)"> Misc</div></div><div id="content1" class="content">',
            '</div><div id="content2" class="content"><form id="version" method="post"> <label for="version">Version: </label></br> <input class="versionInput" type="radio" id="99standard" name="version" value="99standard"',
            '> <label for="99standard">Standard 9x9</label></br> <input class="versionInput" type="radio" id="xSudoku" name="version" value="xSudoku"',
            '> <label for="xSudoku">xSudoku 9x9</label></br> <input class="versionInput" type="radio" id="44standard" name="version" value="44standard"',
            '> <label for="44standard">Standard 4x4</label></br> <input name="type" class="hiddenInputs" value="version"></input> <input name="version" class="hiddenInputs versionHiddenInput" value="',
            '"></input> <input name="task" class="hiddenInputs taskHiddenInput"></input> <button type="button" onclick="submitVersion()">Submit</button></form></div><div id="content3" class="content"> <label for="task">Task: </label></br> <input class="taskInput" type="radio" id="solve" name="task" value="solve"',
            '> <label for="solve">Solve sudoku</label></br> <input class="taskInput" type="radio" id="hint" name="task" value="hint"',
            '> <label for="xSudoku">Provide hint</label></br> <input class="taskInput" type="radio" id="check" name="task" value="check"',
            '> <label for="check">Check sudoku</label></br></div> </label> <label class="control buttons" onclick="sumbitSudoku()"> Solve Sudoku </label></div></div></body></html>',]

    style = '.buttons{cursor:pointer}#header{font-size:10vh;height:10vh;margin:0}#content{display:flex;flex-wrap:wrap}#sudoku{height:80vh;width:80vh;display:grid;grid-template-columns:squarePlaceHolderVhTimes;grid-gap:1vh;background-color:#000;border:solid #ccc 1vh;float:left}.square{position:relative;height:100%;width:100%;display:grid;grid-template-columns:cellPlaceHolderVhTimes;grid-gap:1vh;background-color:silver}.cell{font-size:cellPlaceHolderVh;height:cellPlaceHolderVh;text-align:center;background-color:#fff;border:0}#image,.hiddenInputs{display:none}.control{background-color:#004080;border-radius:5px;border:1px solid #004080;color:#fff}#controls{flex-grow:100;display:grid;grid-template-rows:26vh 26vh 26vh;grid-gap:1vh;white-space:nowrap;font-size:6vh;padding:1vh 0 0 1vh}#controls>.control{height:26vh;width:100%}.tab{width:25%;background-color:ccc}#tab0{width:25%;background-color:#004080}#tabs{width:100%;display:flex;background-color:ccc}.tabOpen{background-color:#fff;color:#000}.content{background-color:#fff;color:#000;display:none;font-size:3vh}#content1{display:block}@media screen and (orientation:portrait){#header{font-size:10vw;height:10vw}#sudoku{height:80vw;width:80vw;grid-template-columns:squarePlaceHolderVwTimes;grid-gap:1vw;border-width:1vw}.square{grid-template-columns:cellPlaceHolderVwTimes;grid-gap:1vw}.cell{font-size:cellPlaceHolderVw;height:cellPlaceHolderVw}#controls{padding-left:0;font-size:6vw;height:6vw;grid-template-rows:26vw 26vw 26vw;grid-gap:1vw;padding:1vw 0 0 1vw}#controls>.control{height:26vw}.content{font-size:3vw}}'

    def sendHtml(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(data)#.encode("utf8"))

    def makeSudoku(self, version, params, opacity):
        sudokuHtml = ''
        if version == '99standard':
            for s in range(9):
                sudokuHtml += '<div id="s' + str(s) + '" class="square">\n'
                for c in range(9):
                    if str(s) + str(c) in params:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="' + params[str(s) + str(c)] + '" class="cell" style="'
                    else:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="" class="cell" style="'

                    if str(s) + str(c) in opacity:
                        sudokuHtml += 'color: rgba(0, 0, 0, ' + str(opacity[str(s) + str(c)]) + ');'

                    sudokuHtml += '">\n'

                sudokuHtml += '</div>\n'

        elif version == 'xSudoku':
            for s in range(9):
                sudokuHtml += '<div id="s' + str(s) + '" class="square">\n'
                for c in range(9):
                    if str(s) + str(c) in params:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="' + params[str(s) + str(c)] + '" class="cell" style="'
                    else:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="" class="cell" style="'

                    if str(s) + str(c) in opacity:
                        sudokuHtml += 'color: rgba(0, 0, 0, ' + str(opacity[str(s) + str(c)]) + ');'

                    if str(s) + str(c) in ['00', '04', '08', '40', '44', '48', '80', '84', '88', '22', '24', '26', '42', '44', '46', '62', '64', '66']:
                        sudokuHtml += 'background-color: beige;'

                    sudokuHtml += '">\n'

                sudokuHtml += '</div>\n'

        elif version == '44standard':
            for s in range(4):
                sudokuHtml += '<div id="s' + str(s) + '" class="square">\n'
                for c in range(4):
                    if str(s) + str(c) in params:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="' + params[str(s) + str(c)] + '" class="cell" style="'
                    else:
                        sudokuHtml += '<input type="text" size="1" maxlength="1" name="' + str(s) + str(c) + '" value="" class="cell" style="'

                    if str(s) + str(c) in opacity:
                        sudokuHtml += 'color: rgba(0, 0, 0, ' + str(opacity[str(s) + str(c)]) + ');'

                    sudokuHtml += '">\n'

                sudokuHtml += '</div>\n'

        else:
           input('error')

        return sudokuHtml

    def applyStylePlaceHolders(self, squareAmount, cellAmount):
            squareSize = (81 - squareAmount)/squareAmount
            cellSize = (squareSize + 1 - cellAmount)/cellAmount

            squarePlaceHolderVhTimes = (str(squareSize) + 'vh ')*squareAmount
            squarePlaceHolderVwTimes = (str(squareSize) + 'vw ')*squareAmount
            cellPlaceHolderVh = (str(cellSize) + 'vh ')
            cellPlaceHolderVw = (str(cellSize) + 'vw ')
            cellPlaceHolderVhTimes = cellPlaceHolderVh*cellAmount
            cellPlaceHolderVwTimes = cellPlaceHolderVw*cellAmount

            return self.style.replace('squarePlaceHolderVhTimes', squarePlaceHolderVhTimes).replace('squarePlaceHolderVwTimes', squarePlaceHolderVwTimes).replace('cellPlaceHolderVhTimes', cellPlaceHolderVhTimes).replace('cellPlaceHolderVwTimes', cellPlaceHolderVwTimes).replace('cellPlaceHolderVh', cellPlaceHolderVh).replace('cellPlaceHolderVw', cellPlaceHolderVw)

    def makeStyle(self, version):
        if version == '99standard':
            versionStyle = self.applyStylePlaceHolders(3, 3)
        elif version == 'xSudoku':
            versionStyle = self.applyStylePlaceHolders(3, 3)
        elif version == '44standard':
            versionStyle = self.applyStylePlaceHolders(2, 2)
        else:
           input('error')

        return versionStyle
        
    def do_GET(self, version = '99standard', task = 'solve', params = {}, opacity = {}, message = ['Enter your sudoku by uploading an image or type directly into the sudoku.', 'Press solve sudoku when ready to.']):
        index = self.html[0]
        index += self.makeStyle(version)
        index += self.html[1]
        index += version
        index += self.html[2]
        index += self.makeSudoku(version, params, opacity)
        index += self.html[3]
        index += version
        index += self.html[4]
        index += '</br>'.join(message)
        index += self.html[5]

        if version == '99standard':
            index += 'checked="checked"'
        index += self.html[6]
        if version == 'xSudoku':
            index += 'checked="checked"'
        index += self.html[7]
        if version == '44standard':
            index += 'checked="checked"'

        index += self.html[8]
        index += version
        index += self.html[9]
            
        if task == 'solve':
            index += 'checked="checked"'
        index += self.html[10]
        if task == 'hint':
            index += 'checked="checked"'
        index += self.html[11]
        if task == 'check':
            index += 'checked="checked"'

        index += self.html[12]

        self.sendHtml(index.encode("utf8"))
            
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        postData = self.rfile.read(content_length)

        if b'image' in postData:
            filename = str(postData.split(b'filename="')[1].split(b'"')[0]).decode("utf-8")
            version = str(postData.split(b'"version"\r\n\r\n')[1].split(b'\r\n')[0]).decode("utf-8")
            task = str(postData.split(b'"task"\r\n\r\n')[1].split(b'\r\n')[0]).decode("utf-8")
            #formatImage(postData)
            self.do_GET(version, task)
        else:
            dataDict = {}
            for datum in postData.decode("utf-8").split('&'):
                key, value = datum.split('=')
                dataDict[str(key)] = str(value)

            if dataDict['type'] == 'sudoku':
                #self.do_GET(dataDict['version'], dataDict['task'], dataDict, ['Sudoku submitted ... Proccesing'])
                self.do_GET(dataDict['version'], dataDict['task'], solveSudoku(dataDict))
            elif dataDict['type'] == 'version':
                self.do_GET(dataDict['version'], dataDict['task'])
            else:
                print('error')

    def parseImage(self, rawData):
        filename = rawData.split(b'filename="')[1].split(b'"')[0]
        data = rawData.split(b'\r\n\r\n')
        with open('test.jpg', 'wb') as f:
            f.write(data[-1])
        return data[-1]

def solveSudoku(rawDataDict):
    return rawDataDict

addr = 'localhost'
port = 8080
serverAddress = (addr, port)

with socketserver.TCPServer(serverAddress, handler) as httpd:
    print(f'Starting httpd server on {addr}:{port}')
    httpd.serve_forever()

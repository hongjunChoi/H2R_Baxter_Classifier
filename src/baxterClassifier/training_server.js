//Lets require/import the HTTP module
var http = require('http');
var express = require('express');
var path = require('path');
var bodyParser = require('body-parser');
var PythonShell = require('python-shell');
var app = express();

app.use(express.static(path.join(__dirname, '/public')));
app.set('views', __dirname + '/public');

app.get('/', function(req, res) {
    console.log("============ HERE ===================\n");
    res.status(200).end("ok");

    var spawn = require('child_process').spawn;
    var py = spawn('python', ['baxterTrain.py']);

    py.on('error', function(err) {
        console.log('ERROR : Failed to start child process.    ', err);
    });

    py.stdout.on('data', function(data) {
        console.log(data.toString());
    });

    py.stderr.on('data', function(data) {
        console.log('ps stderr:  ERROR ', data.toString());
    });

    py.on('close', function(code) {
        if (code !== 0) {
            console.log(`ps process exited with code ${code}`);
        }
        console.log("close .....");
    });


    py.stdout.on('end', function() {
        console.log('done.... ');
    });


    // var PythonShell = require('python-shell');
    // var pyshell = new PythonShell('baxterTrain.py');


    // pyshell.on('message', function(message) {
    //     // received a message sent from the Python script (a simple "print" statement) 
    //     console.log(message);
    // });


    // // end the input stream and allow the process to exit 
    // pyshell.end(function(err) {
    //     if (err) {
    //         console.log("======= ERROR ========");
    //         console.log(err);
    //         return;
    //     }

    //     console.log('finished without error ..');
    // });

});


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: false
}));

//Lets define a port we want to listen to   
const PORT = process.env.PORT || 8080;

//Create a server
var server = http.createServer(app);
server.listen(PORT, function() {
    console.log('Server listening on port ' + PORT);
});


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));

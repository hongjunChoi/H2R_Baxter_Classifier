var http = require('http');
var express  = require('express');
var bodyParser = require('body-parser');
var app= express();

app.use(bodyParser.json());
console.log(__dirname)
app.get('/', function (req, res) {

  res.sendfile(__dirname + '/index.html');
});


app.listen(3000, function() {
	console.log('I\'m Listening...');
})
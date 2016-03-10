var http = require('http');
var express  = require('express');
var app= express();



app.get('/', function (req, res) {

  res.sendfile(__dirname + '/index.html');
});


app.listen(3000, function() {
	console.log('I\'m Listening...');
})
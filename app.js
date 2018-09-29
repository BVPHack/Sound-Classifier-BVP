// Import Express.js JavaScript module into the application
// and creates its variable.
var express = require('express');
var app = express();

// Creates a server which runs on port 3000 and
// can be accessed through localhost:3000
app.listen(3000, function() {
    console.log('server running on port 3000');
} )

// Function callName() is executed whenever
// the URL is of the form localhost:3000/name


function callName() {

    // Use child_process.spawn method from
    // child_process module and assign it
    // to variable spawn
    var spawn = require("child_process").spawn;
     
    // Parameters passed in spawn -
    // 1. type_of_script
    // 2. List containing Path of the script
    //    and arguments for the script

    // E.g.: http://localhost:3000/name?firstname=Mike&lastname=Will
    // So, first name = Mike and last name = Will
    var process = spawn('python',['-u',"./runtime.py",'harshit','kohli'] );

    // Takes stdout data from script which executed
    // with ar9guments and send this data to res object
    var count =0;
    process.stdout.on('data', function(data) {
      console.log(data.toString(),count);
        
      if(data.toString() == "noise")
      {
          console.log('abc',data.toString())
      } 
      
    } )

    
    setTimeout(() => {
        process.kill()
    },20000)
}
callName()
  
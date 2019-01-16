// simplified-parser.js
// transfer quick-draw.ndjson file to json file
// run command:
// node simplified-parser.js "input-ndjson-file-path" "output-json-file-path"

var fs = require('fs');
var ndjson = require('ndjson'); // npm install ndjson
var ndjson_file_name = process.argv[2]; // "./dataset/doodles/ndjson/dog/full_simplified_dog.ndjson" 
var json_file_name = process.argv[3]; // "./dataset/doodles/json/dog/full_simplified_dog.json" 

function parseSimplifiedDrawings(fileName, callback) {
  var drawings = [];
  var fileStream = fs.createReadStream(fileName)
  fileStream
    .pipe(ndjson.parse())
    .on('data', function(obj) {
      drawings.push(obj)
    })
    .on("error", callback)
    .on("end", function() {
      callback(null, drawings)
    });
}

parseSimplifiedDrawings(ndjson_file_name, function(err, drawings) {
  if(err) return console.error(err);
  // drawings.forEach(function(d) {
    // // Do something with the drawing
    // console.log(d.key_id, d.countrycode);
  // })
  // console.log("# of drawings:", drawings);
  var filename = json_file_name;
  fs.writeFileSync(filename, JSON.stringify(drawings));
})

var fs = require('fs');
var ndjson = require('ndjson'); // npm install ndjson

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

parseSimplifiedDrawings("./dataset/doodles/ndjson/dog/full_simplified_dog.ndjson", function(err, drawings) {
  if(err) return console.error(err);
  drawings.forEach(function(d) {
    // Do something with the drawing
    console.log(d.key_id, d.countrycode);
  })
  console.log("# of drawings:", drawings);
  var filename = "./dataset/doodles/json/dog/full_simplified_dog.json";//这里保存
  fs.writeFileSync(filename, JSON.stringify(drawings));//这里保存
})
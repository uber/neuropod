const express = require('express');
const basicAuth = require('express-basic-auth');

const app = express();

// This is temporary (to protect the docs until open sourcing)
app.use(basicAuth({
  challenge: true,
  realm: 'FC19052079C5',
  users: {
      "neuropods": "eNcDQs14jpgLI967m3R2U50f7A2KFuHCoF7nf503N3YpQMZbH4PWRPt07tdxzRZ"
  },
  unauthorizedResponse: 'Restricted area. Please login'
}));
app.use(express.static(__dirname + '/_static'));

app.listen(3000, () => console.log('Listening on port 3000...'));

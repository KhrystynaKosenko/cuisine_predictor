#!/usr/bin/env python

from app import app
host = '10.0.1.9'
# host = '192.168.25.98'
# host='localhost'
app.run(debug=True, host=host)

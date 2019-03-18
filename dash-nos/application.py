# -*- coding: utf-8 -*-
import os
import dash

#from demo import demo_layout, demo_callbacks
from local import local_layout, local_callbacks

app = dash.Dash(__name__)
application = app.server


app.layout = local_layout


local_callbacks(app)


# Load external CSS
external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "//fonts.googleapis.com/css?family=Dosis:Medium",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})


# Running the server
if __name__ == '__main__':
    application.run(host='0.0.0.0', debug=True)


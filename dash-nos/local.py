# -*- coding: utf-8 -*-

"""
The Local version of the app.
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import _pickle as pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import bz2

# load data
umap_df = pd.read_csv("umap_df.csv",index_col='Category')#
print('umap loaded')
model_clone = joblib.load('clf_nos.joblib')
print('model cloned')
#tfidf_vectorizer = joblib.load('tfidf_vectorizer_nos.pkl', mmap_mode = 'r')

with bz2.BZ2File('tfidf_vectorizer_nos.pkl', 'r') as f:
    tfidf_vectorizer = pickle.load(f)

print('tfidf loaded')
encoder = joblib.load('encoder_nos.pkl')
print('encoder loaded')
StandardScaler = joblib.load('scaler_nos.pkl')
print('standarsscaler loaded')

def axis_template_3d( title, type='linear' ):
    return dict(
        showbackground = False,
        backgroundcolor = 'rgb(230, 230, 230)',
        showaxeslabels = False,
        showticklabels = False,
        gridcolor = 'rgb(255, 255, 255)',
        title = title,
        type = type,
        zerolinecolor = 'rgb(255, 255, 255)',
        showspikes=False
        
    )
    
        

def create_figure(selected=False, df = umap_df):

    data = []
    c = {"sport": "#9b59b6", 
         "buitenland": "#3498db",
         "binnenland": "#95a5a6", 
         "politiek": "#e74c3c", 
         "regio": "#34495e", 
         "economie": "#2ecc71"}
     
    if selected:
        df = df
        
    for counter, idx in enumerate(df.groupby(df.index)):

        scatter = go.Scatter3d(
            x=idx[1]['umap_1'],
            y=idx[1]['umap_2'],
            z=idx[1]['umap_3'],
            name = idx[0],
            mode='markers',
            hoverinfo='text',
            text = (idx[1].index + ': ' + idx[1]['Title']),
            marker=dict(
                size=3,
                symbol='circle-dot',
                opacity=0.5,
                color = c[str(idx[0])],            
            )
        )
        data.append(scatter)
    
    # Layout graph
    layout = go.Layout( 
        showlegend = False,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t='15%'
        )
        ,scene = dict(
                xaxis = axis_template_3d(''),
                yaxis = axis_template_3d(''),
                zaxis = axis_template_3d(''),
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.01, y=1.3, z=0.01)))
        ,hovermode = 'closest'
    )

    return {'data': data, 'layout': layout}


# app layout
local_layout = html.Div([ 
    
    # The graph
    dcc.Graph(
        id='umap-3d-plot',
        figure=create_figure(),
        style={'height': '100vh',
               'width': '63%',
               'position': 'relative',
               'float': 'left',
        },
               
    ),
            
            
    html.H3('Nieuwsberichten 2018 NOS.nl', style={
            'width': '55%'
            , 'float':'left'
            , 'position':'absolute'
            , 'margin-top': '2%'
            , 'margin-left': '5%'}),

    # dropdown
    html.Div([

        html.Img(
            src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/NOS_logo.svg/266px-NOS_logo.svg.png",
            style={
                'width': '50%',
                'position': 'relative',
            }
        ),
                    
        # text
        html.H5('Selecteer een rubriek:'),
        
        # dropdown
        dcc.Dropdown(
            options=[
                {'label': c, 'value': c}
                for c in sorted(list(umap_df.index.unique()))
            ],
            value=['binnenland', 'buitenland', 'sport', 'economie', 'politiek', 'regio'],
            multi=True,
            id='category-filter',
        ),
        
        html.P(' '),

        # text
        html.H5('Originele bericht:',id='umap_h4'),
        
         # text            
        html.P('Klik op een bolletje om het nieuwsbericht te bekijken.'),
        
        # print web page
        html.Pre(id='hover-data'),

        # text
        html.H5('Schrijf je eigen nieuwsbericht!'),
        
        # type own artile
        dcc.Textarea(
            placeholder=""" """,
            value="""Typ hier je eigen bericht en laat het algoritme het bijbehorende thema voorspellen...""",
            style={'width': '100%'}, 
            id = 'prediction-input',
        ),
        
        
        html.Button(id='submit-button', n_clicks=0, children='Voorspel!'
                    ,style={'color': 'white'
                            , 'backgroundColor': '#EE3224'
                    },
        ),
                
        # print prediction
        html.Pre(id='prediction'),
        
        ],
        className='six columns', style={
                'float':'right',
                'width': '30%',
                'position': 'relative',
                'borderRadius': '20%',
                'border': 'thin lightgrey solid',
                'padding': '2%',   
                'margin-top': '2%',
                'margin-right': '2%'

        }
    ),
    
],
    className="container",
    id = 'main',
    style={
        'width': '100%',
        'max-width': 'none',
        'font-size': '1.5rem',
        'textAlign': 'center'
    }
)

def local_callbacks(app):
    ## callback for rubriek selectie
    @app.callback(
        Output('umap-3d-plot', 'figure'),
        [Input('category-filter', 'value')])
    def filter(selected_values):
        
        if selected_values:
            figure = create_figure(selected = True, df = umap_df[umap_df.index.isin(selected_values)])
        
        else:
            pass
        
    
        for trace in figure['data']:
            trace['hoverinfo'] = 'text'
    
        return figure
    
 
    #  callback for hover 
    @app.callback(
        Output('hover-data', 'children'),
        [Input('umap-3d-plot', 'clickData')])
    def return_article_name(clickData):
        if clickData is not None:
            
            if 'points' in clickData:
                
                firstPoint = clickData['points'][0]
                if 'pointNumber' in firstPoint:
                    x = firstPoint['x']
                    y = firstPoint['y']
                    z = firstPoint['z']
                    hyperlink = umap_df[(umap_df['umap_1'] ==  x) & (umap_df['umap_2'] == y) & (umap_df['umap_3'] == z)]['Link'][0]
                    
                    web_page = html.Iframe(src='{}'.format(hyperlink),
                    style=dict(border=0), width='100%', height='290',
                    )
                    return web_page



    # callback for prediction
    @app.callback(Output('prediction', 'children'),
                  [Input('submit-button', 'n_clicks')],
                  [State('prediction-input', 'value')])
    def return_prediction(n_clicks, text):
        
        if n_clicks > 0:
            stop_words = ['aan','aangaande','aangezien','al','aldaar','aldus','alhoewel',
             'alias','alle','allebei','alsnog','altijd','altoos','ander','andere','anders',
             'anderszins','behalve','behoudens','beide','beiden','ben','beneden','bent','bepaald','betreffende',
             'bij','binnenin','boven','bovenal','bovendien','bovengenoemd','bovenstaand','bovenvermeld',
             'daar','daarheen','daarin','daarna','daarnet','daarom','daarop','daarvanlangs','dan',
             'dat','de','die','dikwijls','dit','door','doorgaand','dus','echter','eerdat',
             'eerlang','elk','elke','en','enig','enigszins','er','erdoor','even','eveneens',
             'evenwel','gauw','gedurende','gehad','gekund','geleden','gemoeten','gemogen',
             'geweest','gewoon','gewoonweg','haar','had','hadden','hare','heb','hebben','hebt','heeft','hem',
             'hen','het','hierbeneden','hierboven','hij','hoe','hoewel','hun','hunne','ik','ikzelf','in',
             'inmiddels','inzake','is','jezelf','jij','jijzelf','jou','jouw','jouwe','juist','jullie','kan',
             'klaar','kon','konden','krachtens','kunnen','kunt','later','liever','maar','mag','meer','met',
             'mezelf','mij','mijn','mijnent','mijner','mijzelf','misschien','mocht','mochten','moest','moesten',
             'moet','moeten','mogen', 'n', 'na','naar','nadat','net','niet','noch','nog','nogal','nu','of','ofschoon',
             'om','omdat','omstreeks','omtrent','omver','onder','ondertussen','ongeveer','ons',
             'onszelf','onze','ook','op','opnieuw','opzij','over','overigens','pas','precies','reeds',
             'rondom','sedert','sinds','s', 'sindsdien','sommige','spoedig','steeds','tamelijk','tenzij',
             'terwijl','thans','tijdens','toch','toen','toenmaals','toenmalig','tot','totdat','tussen','uit',
             'vaak','van','vandaan','vanuit','vanwege','veeleer','verder','vervolgens','vol',
             'volgens','voor','vooraf','vooral','vooralsnog','voorbij','voordat','voorheen',
             'vrij','vroeg','waar','waarom','wanneer','want','waren','was','wat',
             'wegens','wel', 'we', 'weldra','welk','welke','wie','wiens','wier','wij','wijzelf','zal',
             'ze','zelfs','zichzelf','zij','zijn','zijne','zo','zodra','zonder','zou','zouden','zowat','zulke',
             'zullen','zult', 'zegt','de','en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is',
            'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of',
            'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je',
            'mij', 'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', 'u',
            'want', 'nog', 'zal', 'me', 'zij', 'nu', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al',
            'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hun', 'dus', 'alles',
            'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'kunnen',
            'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'uw', 'iemand', 'geweest',
            'andere']

            tfidf_input = pd.Series(text)
            
            # remove numbers, punctuation and other non-alphabetical items
            tfidf_input = tfidf_input.str.replace("[^a-zA-Z#]", " ")
            tfidf_input = tfidf_input.str.replace("'\'", " ")
            
            # normalize to lowercase
            tfidf_input = tfidf_input.apply(lambda x: x.lower())
            
            # stop word removal
            tfidf_input = tfidf_input.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
            
                    
            input_ready = tfidf_vectorizer.transform(tfidf_input)
            input_ready = pd.DataFrame(input_ready.toarray())

            input_ready = StandardScaler.transform(input_ready)
            input_ready = pd.DataFrame(input_ready)


            pred = model_clone.predict(input_ready)

            pred = encoder.inverse_transform(pred)[0]
            prediction = 'Voorspelling: {}'.format(pred)
            return prediction
            
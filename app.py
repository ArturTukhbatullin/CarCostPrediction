from fastapi import FastAPI
from fastapi import status, Body, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
from Modeling.main import Prediction

app = FastAPI(description='CarCostPredictor', version = '0.0.1')

templates = Jinja2Templates(directory="templates/")

# @app.get('/')
# async def welcome(request:Request) -> HTMLResponse:
#     return templates.TemplateResponse(request, 'main.html', {'message':'welcome'})

@app.get('/')
async def get_params(request:Request) ->HTMLResponse:
    # return {'request':'1'}
    return templates.TemplateResponse(request,'main.html', {'predicted_cost':None, 'request':request})

@app.post('/')
async def post_params(request : Request,
                     name : str=Form(...),\
                     millege : float = Form(...),\
                     engine_volume : float = Form(...),\
                     motor_power  : str = Form(...),\
                     fuel_type  : str = Form(...),\
                     body_type  : str = Form(...),\
                     drive_type  : str = Form(...),\
                     gearbox_type  : str = Form(...),\
                     owners_num  : str = Form(...),\
                     configuration  : str = Form(...),\
                     steering_wheel_type  : str = Form(...),\
                     color  : str = Form(...),\
                     saler_comment  : str = Form(...)
                    ):
    
    params = {'name':[name],
                  'millege':millege,
                  'engine_volume':engine_volume,
                  'motor_power':motor_power,
                  'fuel_type':fuel_type,
                  'body_type':body_type,
                  'drive_type':drive_type,
                  'gearbox_type':gearbox_type,
                  'owners_num':owners_num,
                  'configuration':configuration,
                  'steering_wheel_type':steering_wheel_type,
                  'color':color,
                  'saler_comment':saler_comment
                  }
    data = pd.DataFrame(params)
    
    data['brand'] = data['name'].apply(lambda x: x.split()[0])
    data['model'] = data['name'].apply(lambda x: " ".join(x.split()[1:]))

    p = Prediction('Catboost_model', data)
    predicted_cost = round(p.get_predict(),2)

    return templates.TemplateResponse(
        request,
        'main.html',
        {'predicted_cost':predicted_cost, 
        'request':request,
        "form_data":params}
    )


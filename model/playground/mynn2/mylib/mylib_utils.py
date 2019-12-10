
def exponentially_weighted_averages(Y, B=0.5): # B = 0.1 -> like Y_pred = Y; B = 0.5 -> average values; B = 0.9 -> like straight line  
    import numpy as np
    L = Y.shape[1]
    y_pred_prev = Y[0,0]
    Y_pred = np.zeros((1,L))
    for i in range(L): Y_pred[0,i] = B*y_pred_prev + (1-B)*Y[0,i]
    return Y_pred
# import pandas as pd
# import matplotlib.pyplot as plt
# train = pd.DataFrame({'mass':[154,  260,  148,  218,  150,  142,  148,  316,  148,  148,  152,  198],})
# Y = train['mass'].ravel().reshape(1,len(train['mass'])) # convert dataframe to matrix (1, 12)
# Y_pred = exponentially_weighted_averages(Y, 0.5)
# plt.plot(Y_pred[0,:]); plt.plot(Y[0,:]); plt.show() 

def send_email(request, subject, fromaddr, pwd, toaddr):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    import os
#     codes = []
#     for key in request.COOKIES:
#         if key[0:12] == 'basket_item_':
#             product_name = Product.objects.get(product_code=key[12:]).product_name
#             codes.append('Код:' + str(key[12:]) + '; Количество:' + str(request.COOKIES.get(key)) + '; Название:' + product_name)
    codes = request
    
    #attach test.html
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject

    body = '\n'
    for с in codes:
        body += str(с)
        body += '\n'


    #body = str(codes)
    msg.attach(MIMEText(body, 'plain'))

    #ligin and send
    # gmail - account / security / Unsafe applications / allow
    #server = smtplib.SMTP('smtp.gmail.com', 587) 
    server = smtplib.SMTP('smtp.yandex.ru', 587)
    server.starttls()
    server.login(fromaddr, pwd)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
#send_email(request=['kernel','done'], subject='notify', fromaddr='andrei.pitkevich@yandex.ru', pwd='TempPwdForTensorflowNotifications', toaddr='andrei.pitkevich@yandex.ru')

def unique_count_each_col_value(df, max_crop=0.05):
    nunique = df.nunique(dropna=False).sort_values()
    mask = (nunique.astype(float)/df.shape[0] < max_crop)
    for col in df.loc[:, mask].columns:
        print(df[col].value_counts(dropna=False))


def plot_costs(model): 
    import matplotlib.pyplot as plt
    plt.plot(model._costs); plt.ylabel('cost'); plt.xlabel('iterations (per hundreds)'); plt.title("Learning rate =" + str(model.learning_rate)); plt.show()

def plot_decision_boundary(X, y, predict):
    import numpy as np
    import matplotlib.pyplot as plt
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1 #; print('\n'+'x_min=',x_min,'x_max=',x_max)
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1 #; print('\n'+'y_min=',y_min,'y_max=',y_max)
    h = 0.5# 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #;print('\n'+'xx=',xx.shape,'\n'+'yy=',yy.shape);# xx= (78, 352) # yy= (78, 352)
    x_to_predict = np.c_[xx.ravel(), yy.ravel()].T #;print('x_to_predict=',x_to_predict.shape, type(x_to_predict)) # (2, 27456) <class 'numpy.ndarray'>
    Z = predict(x_to_predict) #;print('Z=',Z,Z.shape) # (1, 27456) # (68640000, 2) before Transpose
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) # BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
    plt.ylabel('x2'); plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral, s=250, edgecolors='b')


def plot_3d(df_real, model, name_x, name_y, name_z, name_pred, only_margin=False):
    from plotly.offline import init_notebook_mode, iplot
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd
    init_notebook_mode(connected=True)
    
    # 1 generate predictions cloud---------------------------------------
    d = model._parameters
    col_x = []; col_y = []; col_z = []; col_pred = []; last_a = None; last_i3=-1;
    for i1 in range(100,1000,20): # mass
        for i2 in np.arange(4,10,0.1): # widths
            for i3 in range(0,200,2): # color_scores
                col_x.append(i1); col_y.append(i2); col_z.append(i3); 
                A = int(np.squeeze(model.predict(np.array([[i1],[i2],[i3]]))))
                if only_margin:
                    if A == last_a or i3<last_i3: col_pred.append(None)
                    else: 
                        if len(col_pred)>0: col_pred[-1] = last_a
                        col_pred.append(A)
                else: col_pred.append(A)
                last_a = A; last_i3=i3
    df_model = pd.DataFrame( {name_x: col_x, name_y: col_y, name_z: col_z, name_pred: col_pred} )

    # 2 delete some values from cloud------------------------------------
    if not only_margin:
        df_model = df_model.sample(frac=1, random_state=0)
        df_model.reset_index(inplace=True, drop=True)
        df_model = df_model.iloc[-500:]
    
    # 3 plot-------------------------------------------------------------
    df0 = df_model[df_model['fruit_label']==0]
    df1 = df_model[df_model['fruit_label']==1]
    df2 = df_real[df_real['fruit_label']==0]
    df3 = df_real[df_real['fruit_label']==1]
    trace0 = go.Scatter3d(x = df0[name_x], y = df0[name_y], z = df0[name_z], mode='markers', 
        marker=dict(color='rgb(100, 0, 0)', size=6,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=1))
    trace1 = go.Scatter3d( x = df1[name_x], y = df1[name_y], z = df1[name_z], mode='markers',
        marker=dict(color='rgb(0, 100, 0)',size=6,symbol='circle',line=dict(color='rgb(204, 204, 204)',width=1),opacity=1))
    trace2 = go.Scatter3d(x = df2[name_x], y = df2[name_y], z = df2[name_z], mode='markers',
        marker=dict(color='rgb(255, 0, 0)',size=12,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=1))
    trace3 = go.Scatter3d(x = df3[name_x], y = df3[name_y], z = df3[name_z], mode='markers',
        marker=dict(color='rgb(0, 255, 0)',size=12,symbol='circle',line=dict(color='rgb(204, 204, 204)',width=1),opacity=1))
    data = [trace0, trace1, trace2, trace3]
    layout = go.Layout(scene = dict(xaxis = dict(title=name_x), yaxis = dict(title=name_y), zaxis = dict(title=name_z),), 
                       width=700, margin=dict(r=20, b=10,l=10, t=10),)
    fig = dict( data=data, layout=layout )
    iplot(fig, filename='elevations-3d-surface')


def plot_boundary_2d_or_3d(X, y, predict, df_real, model, name_x, name_y, name_z, name_pred, only_margin=False):
    import numpy as np
    # if x_train.shape[0]==2: plot_decision_boundary(x_train, y_train_vec.ravel(), lambda x: model.predict(x))
    # if x_train.shape[0]==3 and y_train.shape[0]==2: plot_3d(train, model, name_x='mass',name_y='width',name_z='color_score',name_pred='fruit_label',only_margin=True)
    if X.shape[0]==2: plot_decision_boundary(X, y, predict)
    if X.shape[0]==3 and len(np.unique(y))==2: plot_3d(df_real, model, name_x, name_y, name_z, name_pred, only_margin)



    
#plot_3d(model3, df_real=test, name_x='mass',name_y='width',name_z='color_score',name_pred='fruit_label',only_margin=True)
def plot_3d_one_vs_others(df=None, target_class=0, target_col='fruit_label', name_x='mass',name_y='width',name_z='color_score'):
    from plotly.offline import init_notebook_mode, iplot; import plotly.graph_objs as go; import numpy as np; import pandas as pd
    init_notebook_mode(connected=True)
    # 3 plot-------------------------------------------------------------
    df0 = df[df[target_col]==target_class]
    df1 = df[df[target_col]!=target_class]
    trace0 = go.Scatter3d(x = df0[name_x], y = df0[name_y], z = df0[name_z], mode='markers', marker=dict(color='rgb(100, 0, 0)', size=6,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=1))
    trace1 = go.Scatter3d( x = df1[name_x], y = df1[name_y], z = df1[name_z], mode='markers', marker=dict(color='rgb(0, 100, 0)',size=6,symbol='circle',line=dict(color='rgb(204, 204, 204)',width=1),opacity=1))
    data = [trace0, trace1]#, trace2, trace3]
    layout = go.Layout(scene = dict(xaxis = dict(title=name_x), yaxis = dict(title=name_y), zaxis = dict(title=name_z),), width=700, margin=dict(r=20, b=10,l=10, t=10),)
    fig = dict( data=data, layout=layout )
    iplot(fig, filename='elevations-3d-surface')
#plot_3d_one_vs_others(df=train, target_class=0, target_col='fruit_label', name_x='mass',name_y='width',name_z='color_score')

#plot_3d_one_vs_others(df=train, target_class=0, target_col='fruit_label', name_x='mass',name_y='width',name_z='color_score')
# def plot_decision_boundary_softmax(model, X, y):
#     x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1 #; print('\n'+'x_min=',x_min,'x_max=',x_max)
#     y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1 #; print('\n'+'y_min=',y_min,'y_max=',y_max)
#     h = 0.5# 0.01
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #; print('\n'+'xx=',xx,'\n'+'yy=',yy); print('\n'+'len(xx)=',len(xx),'\n'+'len(yy)=',len(yy)); print('\n'+'len(xx[0])=',len(xx[0]),'\n'+'len(yy[0])=',len(yy[0]));
#     Z = model(np.c_[xx.ravel(), yy.ravel()].T) # (68640000, 2) before Transpose
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG) # BrBG PiYG PRGn PuOr RdBu RdGy RdYlBu RdYlGn Spectral
#     plt.ylabel('x2'); plt.xlabel('x1')
#     plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.PiYG, s=250)




def plot_3d_multiclass(model, df_real, name_x, name_y, name_z, name_pred, only_margin=False):
    from plotly.offline import init_notebook_mode, iplot
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd
    init_notebook_mode(connected=True)
    
    # 1 generate predictions cloud---------------------------------------
    d = model._parameters
    col_x = []; col_y = []; col_z = []; col_pred = []; last_a = None; last_i3=-1;
    for i1 in range(100,1000,20): # mass
        for i2 in np.arange(4,10,0.1): # widths
            for i3 in range(0,200,2): # color_scores
                col_x.append(i1); col_y.append(i2); col_z.append(i3); 
                A = int(np.squeeze(model.predict(np.array([[i1],[i2],[i3]]))))
                if only_margin:
                    if A == last_a or i3<last_i3: col_pred.append(None)
                    else: 
                        if len(col_pred)>0: col_pred[-1] = last_a
                        col_pred.append(A)
                else: col_pred.append(A)
                last_a = A; last_i3=i3
    df_model = pd.DataFrame( {name_x: col_x, name_y: col_y, name_z: col_z, name_pred: col_pred} )

    # 2 delete some values from cloud------------------------------------
    if not only_margin:
        df_model = df_model.sample(frac=1, random_state=0)
        df_model.reset_index(inplace=True, drop=True)
        df_model = df_model.iloc[-500:]
    
    # 3 plot-------------------------------------------------------------
    df0 = df_model[df_model['fruit_label']==0]
    df1 = df_model[df_model['fruit_label']==1]
    df2 = df_real[df_real['fruit_label']==0]
    df3 = df_real[df_real['fruit_label']==1]
    trace0 = go.Scatter3d(x = df0[name_x], y = df0[name_y], z = df0[name_z], mode='markers', 
        marker=dict(color='rgb(100, 0, 0)', size=6,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=1))
    trace1 = go.Scatter3d( x = df1[name_x], y = df1[name_y], z = df1[name_z], mode='markers',
        marker=dict(color='rgb(0, 100, 0)',size=6,symbol='circle',line=dict(color='rgb(204, 204, 204)',width=1),opacity=1))
    trace2 = go.Scatter3d(x = df2[name_x], y = df2[name_y], z = df2[name_z], mode='markers',
        marker=dict(color='rgb(255, 0, 0)',size=12,line=dict(color='rgba(217, 217, 217, 0.14)',width=0.5),opacity=1))
    trace3 = go.Scatter3d(x = df3[name_x], y = df3[name_y], z = df3[name_z], mode='markers',
        marker=dict(color='rgb(0, 255, 0)',size=12,symbol='circle',line=dict(color='rgb(204, 204, 204)',width=1),opacity=1))
    data = [trace0, trace1, trace2, trace3]
    layout = go.Layout(scene = dict(xaxis = dict(title=name_x), yaxis = dict(title=name_y), zaxis = dict(title=name_z),), 
                       width=700, margin=dict(r=20, b=10,l=10, t=10),)
    fig = dict( data=data, layout=layout )
    iplot(fig, filename='elevations-3d-surface')
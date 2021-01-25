import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from apyori import apriori
from django.shortcuts import render
from . import utility
from .forms import AprioriForm, CorrelationForm, MeasurementForm, ClusteringForm, RegressionForm
from scipy.spatial import distance
from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def aprioriAlgorithm(request):
    sheet = None
    results = None
    if request.method == 'POST':
        form = AprioriForm(request.POST, request.FILES)
        if form.is_valid():
            minSupport = form.cleaned_data['soporte_minimo']
            minConfidence = form.cleaned_data['confianza_minima']
            minLift = form.cleaned_data['elevacion_minima']
            minElements = form.cleaned_data['elementos_minimos']

            data = getData(request.FILES['archivo'], 0)
            data.head()
            row, col = data.shape

            registros = []
            for i in range(0, row):
                rowData = []
                for j in range(0, col):
                    if (str(data.values[i, j]) == ''):
                        break
                    rowData.append(data.values[i, j])
                registros.append(rowData)

            reglas = apriori(registros, min_support=minSupport, min_confidence=minConfidence, min_lift=minLift,
                             min_length=minElements)
            resultados = list(reglas)

            results = []
            for item in resultados:
                itemResult = []
                rule = item[0]
                items = [x for x in rule]
                support = item[1]
                confidence = item[2][0][2]
                lift = item[2][0][3]
                itemResult.append(items)
                itemResult.append(utility.truncate(support, 5))
                itemResult.append(utility.truncate(confidence, 5))
                itemResult.append(utility.truncate(lift, 5))
                results.append(itemResult)

            sheet = data.to_html(header=None, max_rows=7)

    form = AprioriForm()
    return render(request, 'ai_algorithms/apriori.html', {'form': form, 'sheet': sheet, 'results': results})


def correlationsAlgorithm(request):
    sheet = None
    response = None
    if request.method == 'POST':
        form = CorrelationForm(request.POST, request.FILES)
        if form.is_valid():
            x = form.cleaned_data['ejeX']
            y = form.cleaned_data['ejeY']
            data = getData(request.FILES['archivo'], 1)
            data.head()

            matriz = data.corr(method='pearson')

            plt.matshow(matriz)
            plt.savefig('ai_algorithms/static/image/correlation/matriz.png')
            plt.close()

            plt.plot(data[x], data[y], 'bo')
            plt.ylabel(y)
            plt.xlabel(x)
            plt.savefig('ai_algorithms/static/image/correlation/correlation.png')
            plt.close()

            sheet = matriz.to_html()

    form = CorrelationForm()
    return render(request, 'ai_algorithms/correlations.html', {'form': form, 'sheet': sheet, 'response': response})


def measurementMethod(request):
    form = MeasurementForm()
    distances = None
    if request.method == 'POST':
        form = MeasurementForm(request.POST, request.FILES)
        if form.is_valid():
            data = getData(request.FILES['archivo'], 2)
            data.drop('ID', 1, inplace=True)
            data.head()
            info = data.to_numpy()
            l = len(info)
            distances = [[0 for i in range(l)] for j in range(l)]

            if(form.cleaned_data['method'] == 'euclidiana'):
                for i in range(l):
                    for j in range(l):
                        distances[i][j] = distance.euclidean(info[i], info[j])
            elif(form.cleaned_data['method'] == 'chevyseb'):
                for i in range(l):
                    for j in range(l):
                        distances[i][j] = distance.chebyshev(info[i], info[j])
            elif (form.cleaned_data['method'] == 'chevyseb'):
                for i in range(l):
                    for j in range(l):
                        distances[i][j] = distance.cityblock(info[i], info[j])
            else:
                for i in range(l):
                    for j in range(l):
                        distances[i][j] = distance.minkowski(info[i], info[j])

            distances = np.tril(distances)
            distances = tabulate(distances, tablefmt='html')

    return render(request, 'ai_algorithms/measurement.html', {'form': form, 'sheet': distances})


def clusteringKmeans(request):
    form = ClusteringForm()
    resultados = None
    pase = None
    if request.method == 'POST':
        form = ClusteringForm(request.POST, request.FILES)
        if form.is_valid():
            data = getData(request.FILES['archivo'], 3)
            matriz = data.corr(method='pearson')

            plt.figure(figsize=(4, 2))
            sb.set(font_scale=0.3)
            heatmap = sb.heatmap(matriz, annot=True)
            figure = heatmap.get_figure()
            figure.savefig('ai_algorithms/static/image/clustering/heatmap.png', dpi=400)

            parametros = form.cleaned_data['parametros']
            parametros = parametros.split()
            for i in range(len(parametros)):
                parametros[i] = int(parametros[i])
            parametros.sort()

            variablesModelo = data.iloc[:, parametros].values
            sheet = pd.DataFrame(variablesModelo)
            SSE = []
            for i in range(2, 16):
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(variablesModelo)
                SSE.append(km.inertia_)

            plt.figure(figsize=(10, 7))
            sb.set(font_scale=1)
            plt.plot(range(2, 16), SSE, marker='o')
            plt.xlabel('Cantidad de clusters *k*')
            plt.ylabel('SSE')
            plt.title('Elbow Method')
            plt.savefig('ai_algorithms/static/image/clustering/plot.png')

            kl = KneeLocator(range(2, 16), SSE, curve="convex", direction="decreasing")

            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(variablesModelo)
            MParticional.predict(variablesModelo)

            data['clusterP'] = MParticional.labels_

            CentroidesP = MParticional.cluster_centers_
            pd.DataFrame(CentroidesP.round(4))

            plt.rcParams['figure.figsize'] = (10, 7)
            plt.style.use('ggplot')
            colores = ['red', 'blue', 'cyan', 'green', 'yellow']
            asignar = []
            for row in MParticional.labels_:
                asignar.append(colores[row])

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(variablesModelo[:, 0], variablesModelo[:, 1], variablesModelo[:, 2], marker='o', c=asignar, s=60)
            ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1], CentroidesP[:, 2], marker='*', c=colores, s=1000)
            plt.savefig('ai_algorithms/static/image/clustering/clusters.png')

            cercanos, _ = pairwise_distances_argmin_min(MParticional.cluster_centers_, variablesModelo)
            pacientes = data['IDNumber'].values
            resultados = []
            for row in cercanos:
                resultados.append(pacientes[row])
            pase = True

    return render(request, 'ai_algorithms/clustering.html', {'form': form, 'pase': pase, 'pacientes': resultados})


def regresionMethod(request):
    form = RegressionForm()
    pase = None
    prediccion = None
    exactitud = None
    if request.method == 'POST':
        form = RegressionForm(request.POST, request.FILES)
        if form.is_valid():
            data = getData(request.FILES['archivo'], 3)

            matriz = data.corr(method='pearson')
            plt.figure(figsize=(4, 2))
            sb.set(font_scale=0.3)
            heatmap = sb.heatmap(matriz, annot=True)
            figure = heatmap.get_figure()
            figure.savefig('ai_algorithms/static/image/regression/heatmap.png', dpi=400)

            bCancer = data.replace({'M': 0, 'B': 1})
            x = np.array(bCancer[['Texture', 'Area', 'Compactness', 'Concavity', 'Symmetry', 'FractalDimension']])

            y = np.array(bCancer[['Diagnosis']])

            plt.figure(figsize=(10, 7))
            plt.scatter(x[:, 0], x[:, 1], c=bCancer.Diagnosis)
            plt.grid()
            plt.xlabel('Texture')
            plt.ylabel('Area')
            plt.savefig('ai_algorithms/static/image/regression/plot.png')

            clasificacion = linear_model.LogisticRegression()
            validation_size = 0.2
            seed = 1234
            x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y,
                                                                                            test_size=validation_size,
                                                                                            random_state=seed,
                                                                                            shuffle=True)

            clasificacion.fit(x_train, y_train)
            probabilidad = clasificacion.predict_proba(x_train)
            predicciones = clasificacion.predict(x_train)

            prediccionesNuevas = clasificacion.predict(x_validation)
            confusion_matrix = pd.crosstab(y_validation.ravel(), prediccionesNuevas, rownames=['Real'],
                                           colnames=['Predicción'])

            prediccionesNuevas = clasificacion.predict(x_validation)
            confusion_matrix = pd.crosstab(y_validation.ravel(), prediccionesNuevas, rownames=['Real'],
                                           colnames=['Predicción'])

            exactitud = clasificacion.score(x_validation, y_validation)

            pacienteTextura= form.cleaned_data['pacienteTextura']
            pacienteArea= form.cleaned_data['pacienteArea']
            pacienteCompacidad= form.cleaned_data['pacienteCompacidad']
            pacienteConcavidad= form.cleaned_data['pacienteConcavidad']
            pacienteSimetria= form.cleaned_data['pacienteSimetria']
            pacienteDimensionFractal= form.cleaned_data['pacienteDimensionFractal']

            nuevoPaciente = pd.DataFrame({'Texture': [pacienteTextura], 'Area': [pacienteArea], 'Compactness': [pacienteCompacidad], 'Concavity': [pacienteConcavidad], 'Symmetry': [pacienteSimetria], 'FractalDimension': [pacienteDimensionFractal]})
            prediccion = clasificacion.predict(nuevoPaciente)
            prediccion = prediccion[0]
            pase = 1

    return render(request, 'ai_algorithms/logisticregresion.html', {'form': form, 'pase': pase, 'prediccion': prediccion, 'exactitud': exactitud})

def getData(archivo, algorithm):
    extension = os.path.splitext(str(archivo))[1]

    if(algorithm == 3):
        return pd.read_csv(archivo)

    if (extension == '.csv'):
        return pd.read_csv(archivo, header=None, keep_default_na=None)
    elif (extension == '.xlsx' or extension == '.xls'):
        return pd.read_excel(archivo, header=None, keep_default_na=None)
    else:
        return pd.read_table(archivo)
        # data = pd.read_table(archivo, sep="\t", header=None, keep_default_na=None)

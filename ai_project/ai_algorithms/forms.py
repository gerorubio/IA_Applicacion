from django import forms


class AprioriForm(forms.Form):
    archivo = forms.FileField(label='Archivo')
    soporte_minimo = forms.FloatField(label='Soporte mínimo', initial=0.0045)
    confianza_minima = forms.FloatField(label='Confianza mínima', initial=0.2)
    elevacion_minima = forms.FloatField(label='Elevación mínima', initial=3)
    elementos_minimos = forms.IntegerField(label='Elementos mínimos', initial=2)

class CorrelationForm(forms.Form):
    archivo = forms.FileField(label='Archivo')
    ejeX = forms.CharField(label="Eje x", max_length=100, initial='Radius')
    ejeY = forms.CharField(label="Eje y", max_length=100, initial='Perimeter')

METHOD_CHOICES = [
    ('euclidiana', 'Distancia Euclidiana'),
    ('chevyseb', 'Distancia de Chevyseb'),
    ('manhattan', 'Distancia de Manhattan'),
    ('minkowsky', 'Distancia de Minkowsky'),
    ]

class MeasurementForm(forms.Form):
    archivo = forms.FileField(label='Archivo')
    method = forms.CharField(label='Método', widget=forms.Select(choices=METHOD_CHOICES))


class ClusteringForm(forms.Form):
    archivo = forms.FileField(label='Archivo')
    parametros = forms.CharField(label="Parametros", initial="3 5 7 8 10 11")

class RegressionForm(forms.Form):
    archivo = forms.FileField(label='Archivo')
    parametros = forms.CharField(label="Parametros", initial="Texture Area Compactness Concavity Symmetry FractalDimension")
    pacienteTextura = forms.FloatField(label="Textura", initial=24.54)
    pacienteArea = forms.FloatField(label="Area", initial=181.0)
    pacienteCompacidad = forms.FloatField(label="Compacidad", initial=0.04362)
    pacienteConcavidad = forms.FloatField(label="Concavidad", initial=0.00000)
    pacienteSimetria = forms.FloatField(label="Simetria", initial=0.1587)
    pacienteDimensionFractal = forms.FloatField(label="Dimensión fractal", initial=1.000000)

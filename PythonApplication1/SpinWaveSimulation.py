#All imports go here
import cmath
import math
import numpy
import decimal
from scipy import integrate

#Defining some variables
centralFreq = math.pi * (2 * 10 * 10 ** 9)
epsZero = 8.85 * 10 ** -12
muZero = 4 * math.pi * 10 ** -7
appliedH = 228 * 79.57747
gamma = 2 * math.pi * 3 * 10 ** 10 * 4 * math.pi * 10 ** -7
satMs = 16500 / (10 ** 4 * muZero) * numpy.sign(appliedH)
gilDamping = 0.008
omegaH = gamma * appliedH
omegaM = gamma * satMs
deltaH = complex(0, centralFreq * gilDamping / gamma)
exchangeA = 2 * 10 ** -7 * 10 ** -4
alphaExchange = 2 * exchangeA / (muZero * satMs ** 2)

#Antenna Geometry
wsignal = 648 * 10 ** -9
wground = 324 * 10 ** -9
wgap = 334 * 10 ** -9

#Metal Characteristics
epsilonSi = 3.8
sigmaFM = 1.7 * 10 ** 7
sigmaRu = 1 / (71 * 10 ** -9)
sigmaPt = 1 / (105 * 10 ** -9)
thicknessSi = 80 * 10 ** -9
thicknessFM = 20 * 10 ** -9
thicknessRu = 5 * 10 ** -9
thicknessPt = 5 * 10 ** -9

#Anisotropy and DMI constants
surface_Ks1 = 0
surface_Ks2 = 0
applied_Hu1 = 2 * surface_Ks1 / (muZero * abs(satMs))
applied_Hu2 = 2 * surface_Ks2 / (muZero * abs(satMs))
pinning_d1y = -1 * applied_Hu2 * muZero * abs(satMs) / (2 * exchangeA)
pinning_d2y = -1 * applied_Hu1 * muZero * abs(satMs) / (2 * exchangeA)
pinning_d1x = 0
pinning_d2x = 0
surface_Ds1 = -1 * (0 * 2.7 * 10 ** -3 * 0.3 * 10 ** -9)
surface_Ds2 = 0
bulk_DD1 = 1j * surface_Ds1 / exchangeA
bulk_DD2 = -1j * surface_Ds2 / exchangeA

def Z0(kk, kks, kkl, kkls):
	numerator = 60 * math.pi
	denominator = math.sqrt(epsilonEff(kk, kks, kkl, kkls)) * ((kellip(kk) / kellip(kks)) + (kellip(kkl) / kellip(kkls)))
	return (numerator / denominator)

def kellip(x):
	#print("x = ", x)
	a = 1
	b = math.sqrt(1 - x**2)
	c = x
	K = math.pi / (2 * a)
	p = 1
	
	while p >= (10 ** (-10)):
		an = (a + b) / 2
		bn = math.sqrt(a*b)
		cn = (a - b) / 2
		#print ("c = ", c)
		#print ("cn = ", cn)
		try:
			p = abs((cn - c) / c)
		except ZeroDivisionError:
			#print("Division by Zero!")
			return K
		#print("p = ", p)
		K = math.pi / (2 * a)
		a = an
		b = bn
		c = cn
	return K
	
def epsilonEff(kk, kks, kkl, kkls):
	numerator = kellip(kks) * kellip(kkl) / (kellip(kk) * kellip(kkls))
	denominator = kellip(kks) * kellip(kkl) / (kellip(kk) * kellip(kkls))
	result = (1 + epsilonSi * numerator) / (1 + denominator)
	return result

def gg(k):
	if abs(k) < 40 / thicknessSi:
		return lambda x : (-1 * math.sinh(thicknessSi * abs(x)) / (epsZero * abs(x) * (math.sinh(thicknessSi * abs(x)) + epsilonSi * math.cosh(thicknessSi * abs(x)))))
	else:
		return lambda x : (-1 / (epsZero * abs(x) * (1 + epsilonSi))) 
	
def Ei(x):
	ei = 1
	for n in range(600,0,-1):
		ei = 1 + n / (x + (n + 1) / ei)
	ei = cmath.exp(-x) / (x + 1 / ei)
	return ei

def Ci(x):
	result = -1 * (Ei(complex(0, x)) + Ei(complex(0, -1*x))) / 2
	return result

def Gsi(k):
	integral = lambda x, t: math.cos(t*x) * gg(k)(x)
	integrated = integrate.quad(integral, 1, 40 / thicknessSi, args =(k), limit = 500)
	result = ((integrated[0] + Ci(40 * k / thicknessSi) / (epsZero * (1 + epsilonSi))) / math.pi)
	return result.real

def Qp(numsignal, numground, meshpts, nummax, deltaW):
	b = numpy.zeros(meshpts)
	a = numpy.zeros((meshpts, meshpts))
	x = numpy.zeros(meshpts)
	g33 = numpy.zeros(nummax)
	g32 = numpy.zeros(2*nummax)
	g31 = numpy.zeros(2*nummax)
	for i in range(numground):
		b[i] = 0
		b[i+numsignal+numground] = 0
	for i in range(numsignal):
		b[i+numground] = 1
	for i in range(nummax):
		g33[i] = Gsi(i * deltaW)
	for i in range(2*nummax):
		g32[i] = Gsi(i * deltaW + wgap)
		g31[i] = Gsi(i * deltaW + 2 * wgap)
	for i in range(numsignal):
		for j in range(numsignal):
			index = abs(i - j)
			a[i+numground, j+numground] = g33[index] * deltaW
	for i in range(numsignal):
		for j in range(numground):
			indexOne = abs(i+numground-1-j)
			indexTwo = abs(numsignal-1-i+j)
			a[i+numground, j] = g32[indexOne] * deltaW
			a[i+numground, j+numsignal+numground] = g32[indexTwo] * deltaW
	for i in range(numground):
		for j in range(numground):
			index = abs(i - j)
			a[i+numground+numsignal, j+numground+numsignal] = g33[index] * deltaW
			a[i, j] = g33[index] * deltaW
	for i in range(numground):
		for j in range(numsignal):
			indexOne = abs(numground-1-i+j)
			indexTwo = abs(i+numsignal-1-j)
			a[i, j+numground] = g32[indexOne] * deltaW
			a[i+numground+numsignal, j+numground] = g32[indexTwo] * deltaW
	for i in range(numground):
		for j in range(numground):
			indexOne = abs(numground-1-i+j)
			indexTwo = abs(i+numground-1-j)
			a[i,j+numground+numsignal] = g31[indexOne] * deltaW
			a[i+numground+numsignal, j] = g31[indexTwo] * deltaW

	print(numpy.linalg.cond(a))
	print(g33)
	print(g32)
	print(g31)
	try:
		ww = numpy.linalg.solve(a, b)
	except numpy.linalg.LinAlgError:
		print('LinAlgError. Couldnt solve matrix equation')
		return 1
	jV = ww.sum()

	for i in range(numground):
		b[i] = 1
		b[i+numsignal+numground] = 1
	for i in range(numsignal):
		b[i+numground] = 0
	try:
		ww = numpy.linalg.solve(a, b)
	except:
		print('LinAlgErro. Couldnt solve matrix equation')
		return 1
	jN = ww.sum()
	Cj = -1*jV / jN
	for i in range(numground):
		b[i] = Cj
		b[i+numsignal+numground] = Cj
	for i in range(numsignal):
		b[i+numground] = 1
	ww = numpy.linalg.solve(a, b)
	Q1 = (ww[0:numground].sum() + ww[numground+numsignal:meshpts].sum()) * deltaW
	Css = Q1 / (1 - Cj)
	gw = numpy.zeros(meshpts + 2)
	gw[0] = Css
	gw[1] = Q1
	for i in range(meshpts):
		gw[i+2] = ww[i]
	return gw



def antennaCalcs():
	kk = wsignal / (wsignal + 2 * wgap)
	kks = math.sqrt(1 - kk ** 2)
	kkl = math.tanh(math.pi / 4 *(wsignal / thicknessSi)) / math.tanh(math.pi / 4 * ((wsignal + 2 * wgap) / thicknessSi))
	kkls = math.sqrt(1 - kkl ** 2)
	epsEff = epsilonEff(kk, kks, kkl, kkls)
	z0 = Z0(kk, kks, kkl, kkls)
	gammafs = complex(0, centralFreq * math.sqrt(epsZero * epsEff * muZero))
	numsignal = 520
	numground = math.floor(numsignal / 2)
	meshpts = numsignal + 2 * numground
	nummax = max(numsignal, numground)
	deltaW = (wsignal + 2 * wground) / meshpts
	Qpresult = Qp(numsignal, numground, meshpts, nummax, deltaW)
	Ycss = complex(0, Qpresult[0])
	Q1 = (Qpresult[2:numground+2].sum() + Qpresult[numground+numsignal+2:meshpts+2].sum()) * deltaW
	Q2 = (deltaW*Qpresult[numground+2:numground+numsignal+2]).sum()
	print(Ycss)
	print(Q1)
	print(Q2)
	linearCapacitance = Ycss * centralFreq
	print(linearCapacitance)
	return linearCapacitance
	

def ww(k, H):
	firstTerm = gamma**2 * H * (H + satMs)
	secondTerm = ((gamma * satMs) ** 2)/4 * (-1 * math.exp(-2 * abs(k) * thicknessFM) + 1)
	result = math.sqrt(firstTerm + secondTerm)
	return result

def Q2(w):
	result = complex(0, w * sigmaRu * muZero)
	return result

def Q4(w):
	result = complex(0, w * sigmaPt * muZero)
	return result

def QQ(w):
	first = abs(cmath.sqrt(Q2(w)))
	second = abs(cmath.sqrt(Q4(w)))
	third = abs(cmath.sqrt(complex(0, w * sigmaFM * muZero)))
	result = max(first, second, third)
	return result

def S0h(K2, k):
	numerator = complex(0, -1 * k * (K2 * cmath.cosh(K2*thicknessRu) + abs(k) * cmath.sinh(K2*thicknessRu)))
	denominator = K2 * (K2 * cmath.sinh(K2 * thicknessRu) + cmath.cosh(K2 * thicknessRu) * abs(k))
	if isinstance(K2 * thicknessRu, complex):
		if (K2 * thicknessRu).real < 600:
			return numerator / denominator
		else:
			return complex(0, -k) / K2
	else:
		if (K2 * thicknessRu) < 600:
			return numerator / denominator
		else:
			return complex(0, -k) / K2

def SLj(K4, k, w):
	if k < 20 * QQ(w):
		numerator = complex(0, k * cmath.exp(-thicknessSi * abs(k)))
		denominator = K4 * cmath.sinh(K4 * thicknessPt) + cmath.cosh(K4 * thicknessPt) * abs(k)
		return numerator / denominator
	else:
		result = complex(numpy.sign(k) * math.exp(-1 * (thicknessSi+thicknessPt) * abs(k)))
		return result

def SLh(K4, k):
	numerator = complex(0, k * (K4 * cmath.cosh(K4 * thicknessPt) + cmath.sinh(K4 * thicknessPt) * abs(k)))
	denominator = K4 * (K4 * cmath.sinh(K4 * thicknessPt) + cmath.cosh(K4 * thicknessPt) * abs(k))
	if isinstance(K4 * thicknessPt, complex):
		if (K4 * thicknessPt).real < 600:
			return numerator / denominator
		else:
			return complex(0, k / K4)
	else:
		if K4 * thicknessPt < 600:
			return numerator / denominator
		else:
			return complex(0, k / K4)

def Cmy(Q, k, wH, w):
	numerator = -1 * (omegaM * Q * k - w * Q ** 2 + w * k ** 2 + w ** 2 * sigmaFM * muZero * 1j)
	denom = numpy.zeros(10, dtype = complex)
	denom[0] = w * sigmaFM * wH * muZero
	denom[1] = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	denom[2] = w * sigmaFM * muZero * omegaM
	denom[3] = -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM
	denom[4] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	denom[5] = -1 * k ** 2 * wH * 1j
	denom[6] = -1 * alphaExchange * k ** 4 * omegaM * 1j
	denom[7] = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	denom[8] = Q ** 2 * omegaM * 1j
	denom[9] = Q ** 2 * wH * 1j
	denominator = denom.sum()
	result = numerator / denominator
	return result

def Chx(Q, k, wH, w):
	num = numpy.zeros(19, dtype = complex)
	num[0] = Q ** 3 * w * k * 1j
	num[1] = 2 * Q ** 2 * alphaExchange * k ** 4 * omegaM * 1j
	num[2] = Q * w ** 2 * sigmaFM * k * muZero * 1j
	num[3] = -1 * Q ** 2 * w * sigmaFM * wH * muZero
	num[4] = -1 * Q ** 2 * w * sigmaFM * muZero * omegaM
	num[5] = 2 * w * sigmaFM * k ** 2 * wH * muZero
	num[6] = w * sigmaFM * k ** 2 * muZero * omegaM
	num[7] = Q ** 4 * alphaExchange * w * sigmaFM * muZero * omegaM
	num[8] = 2 * alphaExchange * w * sigmaFM * alphaExchange * k ** 4 * muZero * omegaM
	num[9] =  -3 * Q ** 2 * alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	num[10] = -1 * Q * w * k ** 3 * 1j
	num[11] = -1 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[12] = -1 * alphaExchange * k ** 6 * omegaM * 1j
	num[13] = -1 * k ** 4 * wH * 1j
	num[14] = -1 * Q ** 4 * alphaExchange * k ** 2 * omegaM * 1j
	num[15] = w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * 1j
	num[16] = w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[17] = alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM * 1j
	num[18] = Q ** 2 * k ** 2 * wH * 1j

	numerator = -1 * num.sum()

	denomOne = k ** 2 - Q ** 2 + w * sigmaFM * muZero * 1j

	denomTwo = numpy.zeros(10, dtype = complex)
	denomTwo[0] = w * sigmaFM * wH * muZero
	denomTwo[1] = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	denomTwo[2] = w * sigmaFM * muZero * omegaM
	denomTwo[3] = -1 * Q ** 2 * alphaExchange * w * muZero * omegaM
	denomTwo[4] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	denomTwo[5] = -1 * k ** 2 * wH * 1j
	denomTwo[6] = -1 * alphaExchange * k ** 4 * omegaM * 1j
	denomTwo[7] = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	denomTwo[8] = Q ** 2 * omegaM * 1j
	denomTwo[9] = Q ** 2 * wH * 1j

	denominator = denomOne * (denomTwo.sum())

	return numerator / denominator

def Chy(Q, k, wH, w):
	num = numpy.zeros(34, dtype = complex)
	num[0] = Q ** 5 * alphaExchange ** 2 * w * sigmaFM * k * muZero * omegaM ** 2
	num[1] = -2 * Q ** 4 * alphaExchange * w * k ** 2 * omegaM * 1j
	num[2] = -1 * Q ** 4 * w * omegaM * 1j
	num[3] = -2 * Q ** 3 * alphaExchange ** 2 * w * sigmaFM * k ** 3 * muZero * omegaM ** 2
	num[4] = -2 * Q ** 3 * alphaExchange * w * sigmaFM * k * wH * muZero * omegaM
	num[5] = -1 * Q ** 3 * alphaExchange * w * sigmaFM * k * muZero * omegaM ** 2
	num[6] = -1 * Q ** 3 * alphaExchange * k ** 3 * omegaM ** 2 * 1j
	num[7] = -1 * Q ** 3 * k * wH * omegaM * 1j
	num[8] = -1 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM * k ** 2 * muZero * omegaM
	num[9] = Q * alphaExchange ** 2 * w * sigmaFM * k ** 5 * muZero * omegaM ** 2
	num[10] = 2 * Q * alphaExchange * w * sigmaFM * k ** 3 * wH * muZero * omegaM
	num[11] = Q * alphaExchange * w * sigmaFM * k ** 3 * muZero * omegaM ** 2
	num[12] = Q * w * sigmaFM * k * wH ** 2 * muZero 
	num[13] = Q * w * sigmaFM * k * wH * muZero * omegaM
	num[14] = -1 * alphaExchange * w ** 3 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM * 1j
	num[15] = alphaExchange * w ** 2 * sigmaFM * k ** 4 * muZero * omegaM 
	num[16] = -1 * w ** 3 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[17] = w ** 2 * sigmaFM * k ** 2 * wH * muZero
	num[18] = -1 * Q ** 4 * w * wH * 1j
	num[19] = -4 * Q ** 3 * alphaExchange * k ** 3 * wH * omegaM * 1j
	num[20] = Q * alphaExchange ** 2 * k ** 7 * omegaM ** 2 * 1j
	num[21] = -1 * Q ** 7 * alphaExchange ** 2 * k * omegaM ** 2 * 1j
	num[22] = -3 * Q ** 3 * alphaExchange ** 2 * k ** 5 * omegaM ** 2 * 1j
	num[23] = -1 * w ** 3 * sigmaFM ** 2 * wH * muZero ** 2 * 1j
	num[24] = -1 * Q ** 3 * k * wH ** 2 * 1j
	num[25] = 2 * Q * alphaExchange * k ** 5 * wH * omegaM * 1j
	num[26] = 2 * Q ** 5 * alphaExchange * k * wH * omegaM * 1j
	num[27] = 3 * Q ** 5 * alphaExchange ** 2 * k ** 3 * omegaM ** 2 * 1j
	num[28] = Q * k ** 3 * wH ** 2 * 1j
	num[29] = Q ** 5 * alphaExchange * k * omegaM ** 2 * 1j
	num[30] = Q ** 2 * w * k ** 2 * wH * 1j
	num[31] = Q ** 2 * alphaExchange * w * k ** 4 * omegaM * 1j
	num[32] = Q ** 2 * alphaExchange * w ** 3 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[33] = Q ** 6 * alphaExchange * w * omegaM * 1j
	
	numerator = -1 * num.sum()

	denom = numpy.zeros(28)
	denom[0] = Q ** 8 * alphaExchange ** 2 * omegaM ** 2
	denom[1] = -4 * Q ** 6 * alphaExchange ** 2 * k ** 2 * omegaM ** 2
	denom[2] = -2 * Q ** 6 * alphaExchange * wH * omegaM
	denom[3] = -2 * Q ** 6 * alphaExchange * omegaM ** 2
	denom[4] = Q ** 4 * alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[5] = 6 * Q ** 4 * alphaExchange ** 2 * k ** 4 * omegaM ** 2
	denom[6] = 6 * Q ** 4 * alphaExchange * k ** 2 * wH * omegaM 
	denom[7] = 4 * Q ** 4 * alphaExchange * k ** 2 * omegaM ** 2
	denom[8] = Q ** 4 * wH ** 2
	denom[9] = 2 * Q ** 4 * wH * omegaM 
	denom[10] = Q ** 4 * omegaM ** 2
	denom[11] = -2 * Q ** 2 * alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM ** 2
	denom[12] = -4 * Q ** 2 * alphaExchange ** 2 * k ** 6 * omegaM ** 2
	denom[13] = -2 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * omegaM
	denom[14] = -2 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[15] = -6 * Q ** 2 * alphaExchange * k ** 4 * wH * omegaM
	denom[16] = -2 * Q ** 2 * alphaExchange * k ** 4 * omegaM ** 2
	denom[17] = -2 * Q ** 2 * k ** 2 * wH ** 2 
	denom[18] = 2 * Q ** 2 * k ** 2 * wH * omegaM
	denom[19] = alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * k ** 4 * muZero ** 2 * omegaM ** 2
	denom[20] = alphaExchange ** 2 * k ** 8 * omegaM ** 2
	denom[21] = 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * wH * muZero ** 2 * omegaM
	denom[22] = 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM ** 2
	denom[23] = 2 * alphaExchange * k ** 6 * wH * omegaM
	denom[24] = w ** 2 * sigmaFM ** 2 * wH ** 2 * muZero ** 2
	denom[25] = 2 * w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * omegaM
	denom[26] = w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[27] = k ** 4 * wH ** 2
	denominator = denom.sum()

	return numerator / denominator

def create_b_var(k, wH, w):
	firstTerm = -3 * alphaExchange ** 2 * k ** 2 * omegaM ** 2
	secondTerm = -alphaExchange * omegaM ** 2
	thirdTerm = -2 * wH * alphaExchange * omegaM
	fourthTerm = -1 * alphaExchange ** 2 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	result = firstTerm + secondTerm + thirdTerm + fourthTerm
	return result

def create_c_var(k, wH, w):
	firstTerm = numpy.zeros(2, dtype = complex)
	firstTerm[0] = 3 * k ** 4 * omegaM ** 2
	firstTerm[1] = 2 * w * sigmaFM * k ** 2 * muZero * omegaM ** 2 * 1j

	secondTerm = numpy.zeros(4, dtype = complex)
	secondTerm[0] = 2 * k ** 2 * omegaM ** 2
	secondTerm[1] = 4 * wH * k ** 2 * omegaM
	secondTerm[2] = 2 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	secondTerm[3] = 2 * w * sigmaFM * wH * muZero * omegaM * 1j

	thirdTerm = wH ** 2
	fourthTerm = -1 * w ** 2
	fifthTerm = omegaM * wH

	result = firstTerm.sum() * alphaExchange ** 2 + secondTerm.sum() * alphaExchange + thirdTerm + fourthTerm + fifthTerm
	return result

def create_d_var(k, wH, w):
	firstTerm = numpy.zeros(2, dtype = complex)
	firstTerm[0] = -1 * k ** 6 * omegaM ** 2
	firstTerm[1] = -1 * w * sigmaFM * k ** 4 * muZero * omegaM ** 2 * 1j

	secondTerm = numpy.zeros(4, dtype = complex)
	secondTerm[0] = -1 * k ** 4 * omegaM ** 2
	secondTerm[1] = -2 * wH * k ** 4 * omegaM
	secondTerm[2] = -2 * w * sigmaFM * wH * muZero * k ** 2 * omegaM * 1j
	secondTerm[3] = -2 * w * sigmaFM * k ** 2 * muZero * omegaM ** 2 * 1j

	thirdTerm = numpy.zeros(7, dtype = complex)
	thirdTerm[0] = w ** 2 * k ** 2
	thirdTerm[1] = -2 * sigmaFM * muZero * w * wH * omegaM * 1j
	thirdTerm[2] = -1 * k ** 2 * wH ** 2
	thirdTerm[3] = -1 * k ** 2 * wH * omegaM
	thirdTerm[4] = -1 * w * sigmaFM * wH ** 2 * muZero * 1j
	thirdTerm[5] = -1 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	thirdTerm[6] = w ** 3 * sigmaFM * muZero * 1j

	result = firstTerm.sum() * alphaExchange ** 2 + secondTerm.sum() * alphaExchange + thirdTerm.sum()
	return result

def create_DD_var(a, b, c, d):
	term = numpy.zeros(8, dtype = complex)
	term[0] = c ** 3 / (27 * a ** 3)
	term[1] = d ** 2 / (4 * a ** 2)
	term[2] = (b ** 3 * d) / (27 * a ** 4)
	term[3] = -1 * (b ** 2 * c ** 2) / (108 * a ** 4)
	term[4] = -1 * (b * c * d) / (6 * a ** 3)
	term[5] = -1 * b ** 3 / (27 * a ** 3)
	term[6] = -1 * d / (2 * a)
	term[7] = b * c / (6 * a ** 2)

	root = cmath.sqrt(term[0:5].sum())
	nonroot = term[5:].sum()
	result = (root + nonroot) ** (1/3)
	return result

def create_Q1_var(a, b, c, DD):
	term = numpy.zeros(3, dtype = complex)
	term[0] = c / (3 * a)
	term[1] = (-1 * b ** 2) / (9 * a ** 2)
	term[2] = -1 * b / (3 * a)
	rootterm = DD - ((term[0] + term[1]) / DD) + term[2]
	#result = cmath.sqrt(rootterm)
	result = cmath.sqrt(DD - (c / (3 * a) - b ** 2 / (9 * a ** 2)) / DD - b / (3 * a))
	return result

def create_Q2_var(a, b, c, DD):
	term = numpy.zeros(3, dtype = complex)
	term[0] = c / (3 * a)
	term[1] = -1 * b ** 2 / (9 * a ** 2)
	term[2] = -1 * b / (3 * a)
	realTerm = (term[0] + term[1]) / (2 * DD) + term[2] - DD / 2
	imagTerm = -1j * (math.sqrt(3) * ((term[0] + term[1]) / DD + DD)) / 2
	result = cmath.sqrt(realTerm + imagTerm)
	return result

def create_Q3_var(a, b, c, DD):
	term = numpy.zeros(6, dtype = complex)
	term[0] = c / (3 * a)
	term[1] = -1 * b ** 2 / (9 * a ** 2)
	term[2] = -1 * b / (3 * a)
	term[3] = -1 * DD / 2
	term[4] = ((1 / 3) / a) * c
	term[5] = -1 * ((1 / 9) / a ** 2) * b ** 2
	realTerm = (term[0] + term[1]) / (2 * DD) + term[2] + term[3]
	imagTerm = (1 / 2) * math.sqrt(3) * ((term[4] + term[5]) / DD + DD) * 1j
	result = cmath.sqrt(realTerm + imagTerm)
	return result

def create_Cmy_vec(Q1, Q2, Q3, k, wH, w):
	temp_Cmy = numpy.zeros(6, dtype = complex)
	temp_Cmy[0] = Cmy(Q1, k, wH, w)
	temp_Cmy[1] = Cmy(Q2, k, wH, w)
	temp_Cmy[2] = Cmy(Q3, k, wH, w)
	temp_Cmy[3] = Cmy(-Q1, k, wH, w)
	temp_Cmy[4] = Cmy(-Q2, k, wH, w)
	temp_Cmy[5] = Cmy(-Q3, k, wH, w)
	return temp_Cmy

def create_Chx_vec(Q1, Q2, Q3, k, wH, w):
	temp_Chx = numpy.zeros(6, dtype = complex)
	temp_Chx[0] = Chx(Q1, k, wH, w)
	temp_Chx[1] = Chx(Q2, k, wH, w)
	temp_Chx[2] = Chx(Q3, k, wH, w)
	temp_Chx[3] = Chx(-Q1, k, wH, w)
	temp_Chx[4] = Chx(-Q2, k, wH, w)
	temp_Chx[5] = Chx(-Q3, k, wH, w)
	return temp_Chx

def create_A_matrix(Q1, Q2, Q3, k, Cmy_vec, Chx_vec, SS0h, SSLh, Rm, Rh):
	temp_A = numpy.zeros((6,6), dtype = complex)
	temp_A[0,0] = Q1 - pinning_d1x + bulk_DD1 * k * Cmy_vec[0]
	temp_A[0,1] = Q2 - pinning_d1x + bulk_DD1 * k * Cmy_vec[1]
	temp_A[0,2] = Q3 - pinning_d1x + bulk_DD1 * k * Cmy_vec[2]
	temp_A[0,3] = -Q1 - pinning_d1x + bulk_DD1 * k * Cmy_vec[3]
	temp_A[0,4] = -Q2 - pinning_d1x + bulk_DD1 * k * Cmy_vec[4]
	temp_A[0,5] = -Q3 - pinning_d1x + bulk_DD1 * k * Cmy_vec[5]
	temp_A[1,0] = (Q1 + pinning_d2x - bulk_DD2 * k * Cmy_vec[0]) * cmath.exp(Q1 * thicknessFM)
	temp_A[1,1] = (Q2 + pinning_d2x - bulk_DD2 * k * Cmy_vec[1]) * cmath.exp(Q2 * thicknessFM)
	temp_A[1,2] = (Q3 + pinning_d2x - bulk_DD2 * k * Cmy_vec[2]) * cmath.exp(Q3 * thicknessFM)
	temp_A[1,3] = (-Q1 + pinning_d2x - bulk_DD2 * k * Cmy_vec[3]) * cmath.exp(-Q1 * thicknessFM)
	temp_A[1,4] = (-Q2 + pinning_d2x - bulk_DD2 * k * Cmy_vec[4]) * cmath.exp(-Q2 * thicknessFM)
	temp_A[1,5] = (-Q3 + pinning_d2x - bulk_DD2 * k * Cmy_vec[5]) * cmath.exp(-Q3 * thicknessFM)
	temp_A[2,0] = Cmy_vec[0] * (Q1 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,1] = Cmy_vec[1] * (Q2 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,2] = Cmy_vec[2] * (Q3 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,3] = Cmy_vec[3] * (-Q1 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,4] = Cmy_vec[4] * (-Q2 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,5] = Cmy_vec[5] * (-Q3 - pinning_d1y) - bulk_DD1 * k
	temp_A[3,0] = (Cmy_vec[0] * (Q1 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(Q1 * thicknessFM)
	temp_A[3,1] = (Cmy_vec[1] * (Q2 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(Q2 * thicknessFM)
	temp_A[3,2] = (Cmy_vec[2] * (Q3 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(Q3 * thicknessFM)
	temp_A[3,3] = (Cmy_vec[3] * (-Q1 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(-Q1 * thicknessFM)
	temp_A[3,4] = (Cmy_vec[4] * (-Q2 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(-Q2 * thicknessFM)
	temp_A[3,5] = (Cmy_vec[5] * (-Q3 + pinning_d2y) + bulk_DD2 * k) * cmath.exp(-Q3 * thicknessFM)
	temp_A[4,0] = Cmy_vec[0] * Rm + Chx_vec[0] * (SS0h + Rh * Q1)
	temp_A[4,1] = Cmy_vec[1] * Rm + Chx_vec[1] * (SS0h + Rh * Q2)
	temp_A[4,2] = Cmy_vec[2] * Rm + Chx_vec[2] * (SS0h + Rh * Q3)
	temp_A[4,3] = Cmy_vec[3] * Rm + Chx_vec[3] * (SS0h + Rh * (-Q1))
	temp_A[4,4] = Cmy_vec[4] * Rm + Chx_vec[4] * (SS0h + Rh * (-Q2))
	temp_A[4,5] = Cmy_vec[5] * Rm + Chx_vec[5] * (SS0h + Rh * (-Q3))
	temp_A[5,0] = (Cmy_vec[0] * Rm + Chx_vec[0] * (SSLh + Rh * Q1)) * cmath.exp(Q1 * thicknessFM)
	temp_A[5,1] = (Cmy_vec[1] * Rm + Chx_vec[1] * (SSLh + Rh * Q2)) * cmath.exp(Q2 * thicknessFM)
	temp_A[5,2] = (Cmy_vec[2] * Rm + Chx_vec[2] * (SSLh + Rh * Q3)) * cmath.exp(Q3 * thicknessFM)
	temp_A[5,3] = (Cmy_vec[3] * Rm + Chx_vec[3] * (SSLh + Rh * (-Q1))) * cmath.exp(-Q1 * thicknessFM)
	temp_A[5,4] = (Cmy_vec[4] * Rm + Chx_vec[4] * (SSLh + Rh * (-Q2))) * cmath.exp(-Q2 * thicknessFM)
	temp_A[5,5] = (Cmy_vec[5] * Rm + Chx_vec[5] * (SSLh + Rh * (-Q3))) * cmath.exp(-Q3 * thicknessFM)

	return temp_A

def create_B_matrix(A):
	temp_B = numpy.zeros((4,4), dtype = complex)
	temp_B[0,0] = A[0,0]*A[1,4]*A[2,5] - A[0,0]*A[1,5]*A[2,4] - A[1,0]*A[0,4]*A[2,5] + A[1,0]*A[0,5]*A[2,4] + A[2,0]*A[0,4]*A[1,5] - A[2,0]*A[0,5]*A[1,4]
	temp_B[0,1] = A[0,1]*A[1,4]*A[2,5] - A[0,1]*A[1,5]*A[2,4] - A[1,1]*A[0,4]*A[2,5] + A[1,1]*A[0,5]*A[2,4] + A[2,1]*A[0,4]*A[1,5] - A[2,1]*A[0,5]*A[1,4]
	temp_B[0,2] = A[0,2]*A[1,4]*A[2,5] - A[0,2]*A[1,5]*A[2,4] - A[1,2]*A[0,4]*A[2,5] + A[1,2]*A[0,5]*A[2,4] + A[2,2]*A[0,4]*A[1,5] - A[2,2]*A[0,5]*A[1,4]
	temp_B[0,3] = A[0,3]*A[1,4]*A[2,5] - A[0,3]*A[1,5]*A[2,4] - A[1,3]*A[0,4]*A[2,5] + A[1,3]*A[0,5]*A[2,4] + A[2,3]*A[0,4]*A[1,5] - A[2,3]*A[0,5]*A[1,4]
	temp_B[1,0] = A[0,0]*A[1,4]*A[3,5] - A[0,0]*A[1,5]*A[3,4] - A[1,0]*A[0,4]*A[3,5] + A[1,0]*A[0,5]*A[3,4] + A[3,0]*A[0,4]*A[1,5] - A[3,0]*A[0,5]*A[1,4]
	temp_B[1,1] = A[0,1]*A[1,4]*A[3,5] - A[0,1]*A[1,5]*A[3,4] - A[1,1]*A[0,4]*A[3,5] + A[1,1]*A[0,5]*A[3,4] + A[3,1]*A[0,4]*A[1,5] - A[3,1]*A[0,5]*A[1,4]
	temp_B[1,2] = A[0,2]*A[1,4]*A[3,5] - A[0,2]*A[1,5]*A[3,4] - A[1,2]*A[0,4]*A[3,5] + A[1,2]*A[0,5]*A[3,4] + A[3,2]*A[0,4]*A[1,5] - A[3,2]*A[0,5]*A[1,4]
	temp_B[1,3] = A[0,3]*A[1,4]*A[3,5] - A[0,3]*A[1,5]*A[3,4] - A[1,3]*A[0,4]*A[3,5] + A[1,3]*A[0,5]*A[3,4] + A[3,3]*A[0,4]*A[1,5] - A[3,3]*A[0,5]*A[1,4]
	temp_B[2,0] = A[0,0]*A[1,4]*A[4,5] - A[0,0]*A[1,5]*A[4,4] - A[1,0]*A[0,4]*A[4,5] + A[1,0]*A[0,5]*A[4,4] + A[4,0]*A[0,4]*A[1,5] - A[4,0]*A[0,5]*A[1,4]
	temp_B[2,1] = A[0,1]*A[1,4]*A[4,5] - A[0,1]*A[1,5]*A[4,4] - A[1,1]*A[0,4]*A[4,5] + A[1,1]*A[0,5]*A[4,4] + A[4,1]*A[0,4]*A[1,5] - A[4,1]*A[0,5]*A[1,4]
	temp_B[2,2] = A[0,2]*A[1,4]*A[4,5] - A[0,2]*A[1,5]*A[4,4] - A[1,2]*A[0,4]*A[4,5] + A[1,2]*A[0,5]*A[4,4] + A[4,2]*A[0,4]*A[1,5] - A[4,2]*A[0,5]*A[1,4]
	temp_B[2,3] = A[0,3]*A[1,4]*A[4,5] - A[0,3]*A[1,5]*A[4,4] - A[1,3]*A[0,4]*A[4,5] + A[1,3]*A[0,5]*A[4,4] + A[4,3]*A[0,4]*A[1,5] - A[4,3]*A[0,5]*A[1,4]
	temp_B[3,0] = A[0,0]*A[1,4]*A[5,5] - A[0,0]*A[1,5]*A[5,4] - A[1,0]*A[0,4]*A[5,5] + A[1,0]*A[0,5]*A[5,4] + A[5,0]*A[0,4]*A[1,5] - A[5,0]*A[0,5]*A[1,4]
	temp_B[3,1] = A[0,1]*A[1,4]*A[5,5] - A[0,1]*A[1,5]*A[5,4] - A[1,1]*A[0,4]*A[5,5] + A[1,1]*A[0,5]*A[5,4] + A[5,1]*A[0,4]*A[1,5] - A[5,1]*A[0,5]*A[1,4]
	temp_B[3,2] = A[0,2]*A[1,4]*A[5,5] - A[0,2]*A[1,5]*A[5,4] - A[1,2]*A[0,4]*A[5,5] + A[1,2]*A[0,5]*A[5,4] + A[5,2]*A[0,4]*A[1,5] - A[5,2]*A[0,5]*A[1,4]
	temp_B[3,3] = A[0,3]*A[1,4]*A[5,5] - A[0,3]*A[1,5]*A[5,4] - A[1,3]*A[0,4]*A[5,5] + A[1,3]*A[0,5]*A[5,4] + A[5,3]*A[0,4]*A[1,5] - A[5,3]*A[0,5]*A[1,4]

	return temp_B

def create_M_vec(A, B, DD, Det):
	M = numpy.zeros(6, dtype = complex)
	M[0] = -(DD / Det) * (B[0,1]*B[1,2]*B[2,3] - B[0,1]*B[1,3]*B[2,2] - B[0,2]*B[1,1]*B[2,3] + B[0,2]*B[2,1]*B[1,3] + B[1,1]*B[0,3]*B[2,2] - B[0,3]*B[1,2]*B[2,1])
	M[1] = (DD / Det) * (B[0,0]*B[1,2]*B[2,3] - B[0,0]*B[1,3]*B[2,2] - B[1,0]*B[0,2]*B[2,3] + B[1,0]*B[0,3]*B[2,2] + B[0,2]*B[2,0]*B[1,3] - B[2,0]*B[0,3]*B[1,2])
	M[2] = -(DD / Det) * (B[0,0]*B[1,1]*B[2,3] - B[0,0]*B[2,1]*B[1,3] - B[0,1]*B[1,0]*B[2,3] + B[0,1]*B[2,0]*B[1,3] + B[1,0]*B[0,3]*B[2,1] - B[1,1]*B[2,0]*B[0,3])
	M[3] = (DD / Det) * (B[0,0]*B[1,1]*B[2,2] - B[0,0]*B[1,2]*B[2,1] - B[0,1]*B[1,0]*B[2,2] + B[0,1]*B[2,0]*B[1,2] + B[1,0]*B[0,2]*B[2,1] - B[0,2]*B[1,1]*B[2,0])
	numerator = -1 * (A[0,0]*A[1,5]*M[0] - A[1,0]*A[0,5]*M[0] + A[0,1]*A[1,5]*M[1] - A[1,1]*A[0,5]*M[1] + A[0,2]*A[1,5]*M[2] - A[1,2]*A[0,5]*M[2] + A[0,3]*A[1,5]*M[3] - A[1,3]*A[0,5]*M[3])
	denominator = A[0,4]*A[1,5] - A[0,5]*A[1,4]
	M[4] = numerator / denominator
	M[5] = -1 * (A[0,0]*M[0] + A[0,1]*M[1] + A[0,2]*M[2] + A[0,3]*M[3] + A[0,4]*M[4]) / A[0,5]
	return M

def create_hxk_var(Chx_vec, M_vec, Q1, Q2, Q3):
	term = numpy.zeros(6, dtype = complex)
	term[0] = Chx_vec[0] * M_vec[0] * cmath.exp(Q1 * thicknessFM)
	term[1] = Chx_vec[1] * M_vec[1] * cmath.exp(Q2 * thicknessFM)
	term[2] = Chx_vec[2] * M_vec[2] * cmath.exp(Q3 * thicknessFM)
	term[3] = Chx_vec[3] * M_vec[3] * cmath.exp(-Q1 * thicknessFM)
	term[4] = Chx_vec[4] * M_vec[4] * cmath.exp(-Q2 * thicknessFM)
	term[5] = Chx_vec[5] * M_vec[5] * cmath.exp(-Q3 * thicknessFM)
	result = term.sum()
	return result

def create_hyl_var(k, K4, hxk):
	numerator = -1 * numpy.sign(k) * 1j * k ** 2 * hxk * cmath.exp(-1 * abs(k) * thicknessFM)
	denominator = k ** 2 * cmath.cosh(K4 * thicknessPt) + K4 * cmath.sinh(K4 * thicknessPt) * abs(k)
	result = numerator / denominator
	return result

def MM(k, wH, w):
	var_K2 = cmath.sqrt(Q2(w) + k ** 2)
	var_K4 = cmath.sqrt(Q4(w) + k ** 2)
	Rm = (k ** 4 - w * sigmaFM * k ** 2 * muZero *1j) / (w ** 2 * sigmaFM ** 2 * muZero ** 2 + k ** 4)
	SS0h = S0h(var_K2, k)
	SSLh = SLh(var_K4, k)
	Rh = (w * sigmaFM * k * muZero + k ** 3 * 1j) / (w ** 2 * sigmaFM ** 2 * muZero ** 2 + k ** 4)
	SSLj = SLj(var_K4, k, w)
	var_b = create_b_var(k, wH, w)
	var_a = alphaExchange ** 2 * omegaM ** 2
	var_c = create_c_var(k, wH, w)
	var_d = create_d_var(k, wH, w)
	var_DD = create_DD_var(var_a, var_b, var_c, var_d)
	var_Q1 = create_Q1_var(var_a, var_b, var_c, var_DD)
	var_Q2 = create_Q2_var(var_a, var_b, var_c, var_DD)
	var_Q3 = create_Q3_var(var_a, var_b, var_c, var_DD)
	vec_Cmy = create_Cmy_vec(var_Q1, var_Q2, var_Q3, k, wH, w)
	vec_Chx = create_Chx_vec(var_Q1, var_Q2, var_Q3, k, wH, w)
	matrix_A = create_A_matrix(var_Q1, var_Q2, var_Q3, k, vec_Cmy, vec_Chx, SS0h, SSLh, Rm, Rh)
	matrix_B = create_B_matrix(matrix_A)
	det_B = numpy.linalg.det(matrix_B)
	var_F = -SSLj
	var_DD = var_F * (matrix_A[0,4] * matrix_A[1,5] - matrix_A[0,5] * matrix_A[1,4])
	vec_M = create_M_vec(matrix_A, matrix_B, var_DD, det_B)
	var_hxk = create_hxk_var(vec_Chx, vec_M, var_Q1, var_Q2, var_Q3)
	var_hyl = create_hyl_var(k, var_K4, var_hxk)
	result_ek = -1 * (centralFreq * muZero * var_hyl) / k
	return result_ek


def main():
	antennaCalcs()
	MM(1,1,1)

if __name__ == '__main__':
	main()
import pandas as pd
import numpy as np

df1 = pd.read_csv("./test_ver3_50000.csv")

df2 = df1.drop(df1.columns[[0,1,6,10,15,17,18,20,24,25,33,34,44]],axis=1) 

df3 = df2.dropna(thresh=32) 

df3['age'] = pd.to_numeric(df3['age'])

df3['age'] = df3['age'].fillna(df3['age'].mean())

df3['age'] = np.where(df3['age']<=18,0, np.where(df3['age']<=30,1,2))

df3['renta'] = pd.to_numeric(df3['renta'])

df3['renta'] = df3['renta'].fillna(0)

df3['renta']=np.where(df3['renta']<=1000,0,
	np.where(df3['renta']<=2000,1,
	np.where(df3['renta']<=5000,2,
	np.where(df3['renta']<=20000,3,
	np.where(df3['renta']<=100000,4,
	np.where(df3['renta']<=500000,5,
	np.where(df3['renta']<=2000000,6,
	np.where(df3['renta']<=10000000,7,8)))))))) 

df3['indrel'] = df3['indrel'].fillna(1)
df3['indrel_1mes'] = df3['indrel_1mes'].fillna(1)
df3['tiprel_1mes'] = df3['tiprel_1mes'].fillna(1) 

df3['ind_nuevo'] = df3['ind_nuevo'].fillna(df3['ind_nuevo'].mean)


from sklearn import preprocessing

df3_1 = df3.select_dtypes(include=[object])

le = preprocessing.LabelEncoder()
df3_1 = df3_1.apply(le.fit_transform)

for i in df3_1:
   df3[i] = df3_1[i]

col_index = ['ind_empleado','pais_residencia','sexo','antiguedad','indresi','indext','canal_entrada','cod_prov','ind_actividad_cliente']

for i in range(0, len(col_index)):
   max = 0
   max_index = 0
   for j in range(0, df3.groupby(col_index[i]).size().shape[0]):
      if col_index[i] == 'cod_prov':
         if df3.groupby('cod_prov').size().index[j] > max:
            max = df3.groupby('cod_prov').size().index[j]
            max_index = j
      elif df3.groupby(col_index[i]).size().index[j] > max:
         max = df3.groupby(col_index[i]).size().index[j]
         max_index = j
   df3[col_index[i]] = df3[col_index[i]].fillna(max_index) 






dfT1 = pd.read_csv("test_1000_2.csv",names = ['fecha_dato1','ncodpers1',
'ind_empleado1','pais_residencia1','sexo1','age1','fecha_alta1','ind_nuevo1',
'antiguedad1','indrel1','ult_fec_cli_1t1','indrel_1mes1','tiprel_1mes1',
'indresi1','indext1','conyuemp1','canal_entrada1','indfall1','tipodom1',
'cod_prov1','nomprov1','ind_actividad_cliente1','renta1','segmento1',
'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1',
'fecha_dato','ncodpers',
'ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo',
'antiguedad','indrel','ult_fec_cli_1t','indrel_1mes','tiprel_1mes',
'indresi','indext','conyuemp','canal_entrada','indfall','tipodom',
'cod_prov','nomprov','ind_actividad_cliente','renta','segmento'])



dfT2 = dfT1.iloc[0:dfT1.shape[0], 48:72]

dfT2 = dfT2.drop(dfT2.columns[[0,1,6,10,15,17,18,20]],axis=1) 



dfT3 = dfT2

dfT3['age'] = pd.to_numeric(dfT3['age'])

dfT3['age'] = dfT3['age'].fillna(dfT3['age'].mean())

dfT3['age'] = np.where(dfT3['age']<=18,0, np.where(dfT3['age']<=30,1,2)) 

dfT3['renta'] = pd.to_numeric(dfT3['renta'])

dfT3['renta'] = dfT3['renta'].fillna(0)

dfT3['renta']=np.where(dfT3['renta']<=1000,0,
	np.where(dfT3['renta']<=2000,1,
	np.where(dfT3['renta']<=5000,2,
	np.where(dfT3['renta']<=20000,3,
	np.where(dfT3['renta']<=100000,4,
	np.where(dfT3['renta']<=500000,5,
	np.where(dfT3['renta']<=2000000,6,
	np.where(dfT3['renta']<=10000000,7,8)))))))) 

dfT3['indrel'] = dfT3['indrel'].fillna(1)
dfT3['indrel_1mes'] = dfT3['indrel_1mes'].fillna(1)
dfT3['tiprel_1mes'] = dfT3['tiprel_1mes'].fillna(1) 

dfT3['ind_nuevo'] = dfT3['ind_nuevo'].fillna(dfT3['ind_nuevo'].mean) 

dfT3_1 = dfT3.select_dtypes(include=[object])

le = preprocessing.LabelEncoder()
dfT3_1 = dfT3_1.apply(le.fit_transform)

for i in dfT3_1:
   dfT3[i] = dfT3_1[i]

col_index = ['ind_empleado','pais_residencia','sexo','antiguedad','indresi','indext','canal_entrada','cod_prov','ind_actividad_cliente']

for i in range(0, len(col_index)):
   max = 0
   max_index = 0
   for j in range(0, dfT3.groupby(col_index[i]).size().shape[0]):
      if col_index[i] == 'cod_prov':
         if dfT3.groupby('cod_prov').size().index[j] > max:
            max = dfT3.groupby('cod_prov').size().index[j]
            max_index = j
      elif dfT3.groupby(col_index[i]).size().index[j] > max:
         max = dfT3.groupby(col_index[i]).size().index[j]
         max_index = j
   dfT3[col_index[i]] = dfT3[col_index[i]].fillna(max_index) 






dfT3_1_1 = dfT3.values






df3_1 = df3.iloc[0:df3.shape[0], 0:16]

df3_1_1 = df3_1.values

df3_2_01 = df3['ind_cco_fin_ult1'].values
df3_2_02 = df3['ind_cder_fin_ult1'].values
df3_2_03 = df3['ind_cno_fin_ult1'].values
df3_2_04 = df3['ind_ctju_fin_ult1'].values
df3_2_05 = df3['ind_ctma_fin_ult1'].values
df3_2_06 = df3['ind_ctop_fin_ult1'].values
df3_2_07 = df3['ind_ctpp_fin_ult1'].values
df3_2_08 = df3['ind_dela_fin_ult1'].values
df3_2_09 = df3['ind_ecue_fin_ult1'].values
df3_2_10 = df3['ind_fond_fin_ult1'].values
df3_2_11 = df3['ind_hip_fin_ult1'].values
df3_2_12 = df3['ind_plan_fin_ult1'].values
df3_2_13 = df3['ind_pres_fin_ult1'].values
df3_2_14 = df3['ind_reca_fin_ult1'].values
df3_2_15 = df3['ind_tjcr_fin_ult1'].values
df3_2_16 = df3['ind_valo_fin_ult1'].values
df3_2_17 = df3['ind_nomina_ult1'].values
df3_2_18 = df3['ind_nom_pens_ult1'].values
df3_2_19 = df3['ind_recibo_ult1'].values


from sklearn import tree

clf = tree.DecisionTreeClassifier()

pred01 = clf.fit(df3_1_1, df3_2_01).predict_proba(dfT3_1_1)
pred02 = clf.fit(df3_1_1, df3_2_02).predict_proba(dfT3_1_1)
pred03 = clf.fit(df3_1_1, df3_2_03).predict_proba(dfT3_1_1)
pred04 = clf.fit(df3_1_1, df3_2_04).predict_proba(dfT3_1_1)
pred05 = clf.fit(df3_1_1, df3_2_05).predict_proba(dfT3_1_1)
pred06 = clf.fit(df3_1_1, df3_2_06).predict_proba(dfT3_1_1)
pred07 = clf.fit(df3_1_1, df3_2_07).predict_proba(dfT3_1_1)
pred08 = clf.fit(df3_1_1, df3_2_08).predict_proba(dfT3_1_1)
pred09 = clf.fit(df3_1_1, df3_2_09).predict_proba(dfT3_1_1)
pred10 = clf.fit(df3_1_1, df3_2_10).predict_proba(dfT3_1_1)
pred11 = clf.fit(df3_1_1, df3_2_11).predict_proba(dfT3_1_1)
pred12 = clf.fit(df3_1_1, df3_2_12).predict_proba(dfT3_1_1)
pred13 = clf.fit(df3_1_1, df3_2_13).predict_proba(dfT3_1_1)
pred14 = clf.fit(df3_1_1, df3_2_14).predict_proba(dfT3_1_1)
pred15 = clf.fit(df3_1_1, df3_2_15).predict_proba(dfT3_1_1)
pred16 = clf.fit(df3_1_1, df3_2_16).predict_proba(dfT3_1_1)
pred17 = clf.fit(df3_1_1, df3_2_17).predict_proba(dfT3_1_1)
pred18 = clf.fit(df3_1_1, df3_2_18).predict_proba(dfT3_1_1)
pred19 = clf.fit(df3_1_1, df3_2_19).predict_proba(dfT3_1_1)



p01=pred01.tolist()
p02=pred02.tolist()
p03=pred03.tolist()
p04=pred04.tolist()
p05=pred05.tolist()
p06=pred06.tolist()
p07=pred07.tolist()
p08=pred08.tolist()
p09=pred09.tolist()
p10=pred10.tolist()
p11=pred11.tolist()
p12=pred12.tolist()
p13=pred13.tolist()
p14=pred14.tolist()
p15=pred15.tolist()
p16=pred16.tolist()
p17=pred17.tolist()
p18=pred18.tolist()
p19=pred19.tolist()


for i in range(0,1000):
	del(p01[i][0])
	p01[i].append('ind_cco_fin_ult1')
        del(p02[i][0])
        p02[i].append('ind_cder_fin_ult1')
        del(p03[i][0])
        p03[i].append('ind_cno_fin_ult1')
        del(p04[i][0])
        p04[i].append('ind_ctju_fin_ult1')
        del(p05[i][0])
        p05[i].append('ind_ctma_fin_ult1')
        del(p06[i][0])
        p06[i].append('ind_ctop_fin_ult1')
        del(p07[i][0])
        p07[i].append('ind_ctpp_fin_ult1')
        del(p08[i][0])
        p08[i].append('ind_dela_fin_ult1')
        del(p09[i][0])
        p09[i].append('ind_ecue_fin_ult1')
        del(p10[i][0])
        p10[i].append('ind_fond_fin_ult1')
        del(p11[i][0])
        p11[i].append('ind_hip_fin_ult1')
        del(p12[i][0])
        p12[i].append('ind_plan_fin_ult1')
        del(p13[i][0])
        p13[i].append('ind_pres_fin_ult1')
        del(p14[i][0])
        p14[i].append('ind_reca_fin_ult1')
        del(p15[i][0])
        p15[i].append('ind_tjcr_fin_ult1')
        del(p16[i][0])
        p16[i].append('ind_valo_fin_ult1')
        del(p17[i][0])
        p17[i].append('ind_nomina_ult1')
        del(p18[i][0])
        p18[i].append('ind_nom_pens_ult1')
	del(p19[i][0])
	p19[i].append('ind_recibo_ult1')

zero_proba=['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cder_fin_ult1',
'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_hip_fin_ult1','ind_pres_fin_ult1','ind_viv_fin_ult1']

for i in range(0,1000):
	r_1=[p01[i],p03[i],p04[i],p05[i],p06[i],p08[i]
	,p09[i],p10[i],p12[i],p14[i],p15[i],p16[i],p17[i]
	,p18[i],p19[i]]
	r_2 = sorted(r_1,reverse=True, key = lambda x:x[0])
	data = {'01':[r_2[0][1]],
		'02':[r_2[1][1]],
		'03':[r_2[2][1]],
                '04':[r_2[3][1]],
                '05':[r_2[4][1]],
                '06':[r_2[5][1]],
                '07':[r_2[6][1]],
                '08':[r_2[7][1]],
                '09':[r_2[8][1]],
                '10':[r_2[9][1]],
                '11':[r_2[10][1]],
                '12':[r_2[11][1]],
		'13':[r_2[12][1]],
		'14':[r_2[13][1]],
		'15':[r_2[14][1]],
                '16':[zero_proba[0]],
                '17':[zero_proba[1]],
                '18':[zero_proba[2]],
                '19':[zero_proba[3]],
                '20':[zero_proba[4]],
                '21':[zero_proba[5]],
                '22':[zero_proba[6]],
                '23':[zero_proba[7]],
                '24':[zero_proba[8]]
					}
	result=pd.DataFrame(data,index=[dfT1['ncodpers'][i]])
	result.to_csv('result.csv',mode='a',header=False)
	


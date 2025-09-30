for q in range (len(qamples)):
  sample_Matrices = [qamples[q].iloc[int(m*len(qamples[q])/2):int((m+1)*(len(qamples[q])/2))].\
            reset_index(drop=True) for m in range(2)]
  sample_MatricesR = [qamplesr[q].iloc[int(m*len(qamplesr[q])/2):int((m+1)*(len(qamplesr[q])/2))].\
            reset_index(drop=True) for m in range(2)]
  
  sample_Matrices_dic = create_dict(letters, sample_Matrices)
  sample_MatricesR_dic = create_dict(letters, sample_MatricesR)
  
  mixed_Matrices = []
  mm_names = []
  mixed_MatricesR = []
  for sm in range (0,len(sample_Matrices)):
    for sm1 in range (0,len(sample_Matrices)):
      if sm == sm1:
        continue
      else:
        for c in sample_Matrices[sm]:
          mixed_Matrices.append(sample_Matrices[sm].copy())
          mixed_Matrices[len(mixed_Matrices)-1][c]=sample_Matrices[sm1].copy()[c]
          mm_names.append(str(letters[sm] + letters[sm1] + str(c+1)))
          mixed_MatricesR.append(sample_MatricesR[sm].copy())
          mixed_MatricesR[len(mixed_MatricesR)-1][c]=sample_MatricesR[sm1].copy()[c]
          
  mixed_Matrices_dic = create_dict(mm_names, mixed_Matrices)
  mixed_MatricesR_dic = create_dict(mm_names, mixed_MatricesR)
  
  matrices_dic = {**sample_Matrices_dic, **mixed_Matrices_dic}
  matricesR_dic = {**sample_MatricesR_dic, **mixed_MatricesR_dic}
  
  names1 = []
  values1R = []
  values1 = []
  names2 = []
  values2 = []
  values2R = []
  for f in functions:
    for sq, zq in mixed_Matrices_dic.items():
      names1.append(f.__name__+str(sq))
      values1.append(f(zq))
    for sqR, zqR in mixed_MatricesR_dic.items():
      values1R.append(f(zqR))
      
    for sM, zM in matrices_dic.items():
      names2.append(f.__name__+str(sM))
      values2.append(f(zM))
    for sMR, zMR in matricesR_dic.items():
      values2R.append(f(zMR))
          
  f_MM_dic = create_dict(names1, values1)
  f_matrices_dic = create_dict(names2, values2)
  f_MMR_dic = create_dict(names1, values1R)
  f_matricesR_dic = create_dict(names2, values2R)
  
  Check=[]
  CheckR=[]
  CheckName = []
  Check3=[]
  Check3R=[]
  Check3Name = []
  for f in functions:
    for j in range(1,k+1):
      difference = []
      difference3 = []
      differenceR = []
      difference3R = []
      for mk, mz in f_matrices_dic.items():
        if mk[0:2]==f.__name__:
          validkeys = []
          validkeys3 = []
          for fk1 in f_MM_dic.keys():
            if len(mk)==3 and mk[2]=='a': 
              if fk1[0:3]==mk[0:3] and fk1[-1]==str(j):
                validkeys.append(fk1)
              if fk1[0:2]==mk[0:2] and fk1[2]!=mk[2] and fk1[-1]==str(j):
                validkeys3.append(fk1)
          z1 = dict(filter(lambda i:i[0] in validkeys, f_MM_dic.items()))
          z3 = dict(filter(lambda i3:i3[0] in validkeys3, f_MM_dic.items()))
          for zk, zv in z1.items():
            difference.append(0.5*(((mz-zv)**2).mean())/mz.var())
          for zk3, zv3 in z3.items():
            difference3.append(((mz*zv3).mean()-mz.mean()**2)/mz.var())
      Check.append(sum(difference)/len(difference))
      CheckName.append('Jansen'+ str(f.__name__) +'ST'+str(j))
      Check3.append(sum(difference3)/len(difference3))
      Check3Name.append('Sobol'+ str(f.__name__) +'S'+str(j))
      for mkR, mzR in f_matricesR_dic.items():
        if mkR[0:2]==f.__name__:
          validkeysR = []
          validkeys3R = []
          for fk1R in f_MMR_dic.keys():
            if len(mkR)==3 and mkR[2]=='a': 
              if fk1R[0:3]==mkR[0:3] and fk1R[-1]==str(j):
                validkeysR.append(fk1R)
              if fk1R[0:2]==mkR[0:2] and fk1R[2]!=mkR[2] and fk1R[-1]==str(j):
                validkeys3R.append(fk1R)
          z1R = dict(filter(lambda iR:iR[0] in validkeysR, f_MMR_dic.items()))
          z3R = dict(filter(lambda iR3:iR3[0] in validkeys3R, f_MMR_dic.items()))
          for zkR, zvR in z1R.items():
            differenceR.append(0.5*(((mzR-zvR)**2).mean())/mzR.var())
          for zk3R, zv3R in z3R.items():
            difference3R.append(((mzR*zv3R).mean()-mzR.mean()**2)/mzR.var())  
      CheckR.append(sum(differenceR)/len(differenceR))
      Check3R.append(sum(difference3R)/len(difference3R))
  Check_dic = create_dict(CheckName, Check)
  Check3_dic = create_dict(Check3Name, Check3)
  CheckR_dic = create_dict(CheckName, CheckR)
  Check3R_dic = create_dict(Check3Name, Check3R)
  
  CheckMAEs = []
  CheckMAEsR = []
  CheckMAENames = []
  CheckMAEsF = []
  CheckMAEsFR = []
  CheckMAEFNames = []
  for ae, av in AE_dic.items():
    for Lk, Lv in Check_dic.items():
      if ae[-5:]==Lk[-5:]:
        CheckMAEs.append(abs(Lv-av))
        CheckMAENames.append('CheckMAE'+ str(ae[2:4]) + str(ae[-1]))
    for LkR, LvR in CheckR_dic.items():
      if ae[-5:]==LkR[-5:]:
        CheckMAEsR.append(abs(LvR-av))
  for af, afv in AEF_dic.items():
    for Lk3, Lv3 in Check3_dic.items():
      if af[-4:]==Lk3[-4:]:
        CheckMAEsF.append(abs(Lv3-afv))
        CheckMAEFNames.append('CheckMAE'+ str(af[2:4]) + str(af[-1]))
    for Lk3R, Lv3R in Check3R_dic.items():
      if af[-4:]==Lk3R[-4:]:
        CheckMAEsFR.append(abs(Lv3R-afv))
  CheckMAEs_dic = create_dict(CheckMAENames, CheckMAEs)
  CheckMAEsF_dic = create_dict(CheckMAEFNames, CheckMAEsF)
  CheckMAEsR_dic = create_dict(CheckMAENames, CheckMAEsR)
  CheckMAEsFR_dic = create_dict(CheckMAEFNames, CheckMAEsFR)
  
  CheckMAE = []
  CheckMAER = []
  CheckMAE_name = []
  CheckMAEF = []
  CheckMAEFR = []
  for f in functions:
    validkeys2 = []
    validkeys4 = []
    validkeys2R = []
    validkeys4R = []
    for Lmk, Lmv in CheckMAEs_dic.items():
      if Lmk[-3:-1]==f.__name__:
        validkeys2.append(Lmk)
    for Fmk, Fmv in CheckMAEsF_dic.items():
      if Fmk[-3:-1]==f.__name__:
        validkeys4.append(Fmk)
    z2 = dict(filter(lambda i2:i2[0] in validkeys2, CheckMAEs_dic.items()))
    z4 = dict(filter(lambda i4:i4[0] in validkeys4, CheckMAEsF_dic.items()))
    CheckMAE.append(sum(z2.values())/len(z2))
    CheckMAE_name.append('CheckMAE'+f.__name__)
    CheckMAEF.append(sum(z4.values())/len(z4))
    for LmkR, LmvR in CheckMAEsR_dic.items():
      if LmkR[-3:-1]==f.__name__:
        validkeys2R.append(LmkR)
    for FmkR, FmvR in CheckMAEsFR_dic.items():
      if FmkR[-3:-1]==f.__name__:
        validkeys4R.append(FmkR)
    z2R = dict(filter(lambda i2R:i2R[0] in validkeys2R, CheckMAEsR_dic.items()))
    z4R = dict(filter(lambda i4R:i4R[0] in validkeys4R, CheckMAEsFR_dic.items()))
    CheckMAER.append(sum(z2R.values())/len(z2R))
    CheckMAEFR.append(sum(z4R.values())/len(z4R))
  CheckMAE_dic = create_dict(CheckMAE_name, CheckMAE)
  CheckMAEF_dic = create_dict(CheckMAE_name, CheckMAEF)
  CheckMAER_dic = create_dict(CheckMAE_name, CheckMAER)
  CheckMAEFR_dic = create_dict(CheckMAE_name, CheckMAEFR)
  
  p_sample.append(CheckMAE_dic)
  p_sample_name.append(2**(q+2))
  f_sample.append(CheckMAER_dic)
  p_sampleR.append(CheckMAEF_dic)
  f_sampleR.append(CheckMAEFR_dic)
p_sample_dic = create_dict(p_sample_name, p_sample)
f_sample_dic = create_dict(p_sample_name, f_sample)
p_sampleR_dic = create_dict(p_sample_name, p_sampleR)
f_sampleR_dic = create_dict(p_sample_name, f_sampleR)

Checkq = pd.DataFrame.from_dict(p_sample_dic)
Checkf = pd.DataFrame.from_dict(f_sample_dic)
CheckR = pd.DataFrame.from_dict(p_sampleR_dic)
CheckfR = pd.DataFrame.from_dict(f_sampleR_dic)

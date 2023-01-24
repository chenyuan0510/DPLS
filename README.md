# DPLS
Quantifying the predictive ability of variable X  to variable Y

### Guides
#### Quick start:
1）Running DeepID requires the python (3.7 version or later) runtime environment; 
2）Make sure that extension package including Numpy, Pandas, scikit-learn and scipy have installed for current python environment;
3）Download DPLS.py to the running directory;
4）The command for  calculating DPLS-R² between dependent variable(Y) and independent variable(X) is:
 ``` 
 from DPLS import dpls_score
 dpls_score(X,Y)
 ```
5The command for  calculating DPLS-R² between dependent variable(Y) and  all variables in the data is:
   from DPLS import dpls
     dpls(data,Y)
6.The command for  calculating DPLS-R² between between all variables in the data is:
     from DPLS import dpls
     dpls(data)
6.If there are many variables and parallel computation is allowed,download DPLS_parallel.py to the running directory;
7.In addition to these extension package,make sure that swifter,joblibhave installed for current python environment
8.The command for  calculating DPLS-R² between dependent variable(Y) andand  all variables in the data is:
    from DPLS_parallel import dpls
    dpls(data,Y)
9.The command for  calculating DPLS-R² between between all variables in the data is:
     from DPLS_parallel import dpls
     dpls(data,njob=n)

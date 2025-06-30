# Pkg
#isplay(countmap(df[!, :sirinaLatice]))
#df[ismissing.(df[!, :sirinaLatice]),:sirinaLatice] .=mode(skipmissing(df[!,:sirinaLatice]))
#plot = scatter(df.Visina, df.Tezina, title="Tezina-Visina",xlabel="Visina", ylabel="Tezina", legend=true)
#savefig(plot,"visinaTezina.html")
#filter!(row -> row.Tezina<=1500, df)
#select!(df,Not(:Duzina)
using Statistics
using StatsModels
using StatsBase
using Printf
using LIBSVM
using DataFrames
using CSV
using Lathe
using Plots, StatsPlots 
using ROC
using MLBase
using Random

# Ucitavanje podataka!
df = DataFrame(CSV.File("cvijece.csv"))
display(describe(df))

X = Matrix(df[:, 2:5])' #formula svi nezavisni podaci
y = df.podvrsta #zavisna promjenljiva

k = 5
a = collect(Kfold(length(df.ID), k))
averageAbsMeanErrorTest = 0.0
for i in 1:k

    Xtrain = df[a[i], : ]
    Xtest = df[setdiff(1:end, a[i]), :]
    Xtrain = X[:, 1:2:end]
    Xtest  = X[:, 2:2:end]
    ytrain = y[1:2:end]#trenirani podaci 
    ytest  = y[2:2:end]

    svmKlasifikator = svmtrain(Xtrain, ytrain, cost = 1.0 )

    Y, decision_values = svmpredict(svmKlasifikator, Xtest);

end
yTestClass = repeat(0:0, length(Y))

for i in 1:length(yTestClass)
    if Y[i] == "Setosa"
        yTestClass[i] = 1
    else
        yTestClass[i] = 0
    end
end
#display(yTestClass)
FPTest = 0
FNTest = 0
TPTest = 0
TNTest = 0

for i in 1:length(yTestClass)
    if Y[i] == "Setosa" && yTestClass[i] == 1
        global TPTest+=1
    elseif Y[i] == "Virginica" && yTestClass[i] == 0
        global TNTest +=1
    elseif Y[i] == "Virginica" && yTestClass[i] == 1
        global FPTest +=1
    elseif Y[i] == "Setosa" && yTestClass[i] == 0
        global FNTest +=1
    end    
end


preciznost = (TPTest+TNTest)/(TPTest+TNTest+FPTest+FNTest)*100
osetljivost = (TPTest)/(TPTest+FNTest)*100
specificnost = (TNTest)/(TNTest+FPTest)*100
    
println("preciznost:  $preciznost %")
println("osetljivost : $osetljivost %")
println("specificnost : $specificnost %")

rocTest = ROC.roc(decision_values, y, "Setosa")
aucTest = AUC(rocTest)

if aucTest>0.9
    println("Klasifikator je jako dobar")
elseif aucTest>0.8
    println("Klasifikator je veoma dobar")
elseif aucTest>0.7
    println("Klasifikator je dosta dobar")
elseif aucTest >0.5
    println("Klasifikator je relativno dobar")
else
    println("Klasifikator je los")
end

plot(rocTest)


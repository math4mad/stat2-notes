using  CodecBzip2,Pipe,FileIO
using  RData,DataFrames,PrettyTables
using  GLMakie
using  HypothesisTests,StatsBase
using  GLM,AnovaGLM,AnovaBase

"""
    load_rda(str::AbstractString)
    加载 Stat2 rda  dataset
"""
function load_rda(str::AbstractString)
 df=load("../Stat2Data/$str.rda")
 return df["$str"]
end

"""
    list_features(df::AbstractDataFrame)

列出 dataframe  names
"""
list_features(df::AbstractDataFrame) = show(names(df))::Nothing

"""
    peek(df::AbstractDataFrame)

show  first 5 row of  dataframe
"""
function peek(df::AbstractDataFrame)
    first(df, 5)
end


Base.@kwdef struct  Stat2Table
    page::Int
    name::AbstractString
    question:: AbstractString
    feature::Vector{Union{AbstractString,Symbol}}
end


"""
    plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String,save::Bool=false)
    使用两个 feature 绘制散点图
    ## Params
    1. data::DataFrame
    2. xlabel::  x feature
    3. ylabel::  y feature
    4. save:: 是否保存图片 默认 false
    ## 返回值
       fig,ax
"""
function plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String,save::Bool=false)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax.title="$(xlabel)-$(ylabel)-scatter"
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.5)
    scatter!(ax,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    save&&save("$(xlabel)-$(ylabel)-scatter.png",fig)
    return (fig,ax)
end

"""
    summary_df(gdf::GroupedDataFrame,feature::Union{String,Symbol})

分组 dataframe summary 选择 feature

return n , mean,stddev
"""
function summary_df(gdf::GroupedDataFrame,feature::Union{String,Symbol})
    return combine(gdf,nrow=>:n ,feature=>mean=>:Mean,feature=>std=>:Stddev)
end


"""
    plot_linreg_residuals(model::StatsModels,data::AbstractDataFrame)

    plot GLM LinearRegression model residuals results with  two column dataframe data
"""
function plot_linreg_residuals(model::StatsModels.TableRegressionModel,data::AbstractDataFrame)
    @assert size(data,2)==2
    resis=residuals(model)
    coefs=coef(model)
    labels=names(data)
    y_hat=predict(model)
    fig=Figure(resolution=(1200,400))
    axs=[Axis(fig[1,i]) for i in 1:3]
    axs[1].xlabel=labels[1];axs[1].ylabel=labels[2]
    axs[2].xlabel="Residuals";axs[2].ylabel="Frequency"
    axs[3].xlabel="Fit Value";axs[3].ylabel="Residuals"
    scatter!(axs[1],eachcol(data)...;marker='o',color=:red)
    ablines!(axs[1],coefs..., linewidth=2, color=:blue)
    hist!(axs[2],resis;bins=15,color = :gray, strokewidth = 1, strokecolor = :black)
    scatter!(axs[3],y_hat,resis;marker='o',color=:red)
    fig
end

"""
    plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String,save::Bool=false)
    使用两个 feature 绘制散点图
    ## Params
    1. data::DataFrame
    2. xlabel::  x feature
    3. ylabel::  y feature
    4. save:: 是否保存图片 默认 false
    ## 返回值
       fig,ax
"""
function plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String,save::Bool=false)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax.title="$(xlabel)-$(ylabel)-scatter"
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.5)
    scatter!(ax,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    #save==true ? save("$(xlabel)-$(ylabel)-scatter.png",fig) : 
    return (fig,ax)
end


"""
    plot_fitline_and_residual(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    绘制回归模型图: fig[1,1] 散点+拟合线, fig[1,2] 预测残差图

 ## Params
    1. data   df
    2. xlabel 预测变量  
    3. ylabel  响应变量
    4. model   回归模型
 ##  返回值
    fig  Makie 对象
"""
function plot_fitline_and_residual(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    y_hat=@pipe select(df,xlabel)|>predict(model,_)|>round.(_,digits=2)
    res=residuals(model)
    fig=Figure(resolution=(800,300))
    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    #ax1.title="$(xlabel)-$(ylabel)-scatter"
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.5)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.5)
    scatter!(ax1,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    lines!(ax1,data[!,xlabel],y_hat,label="fit_line")
    stem!(ax2,res)
    return fig
end

"""
    plot_lm_res(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    绘制回归模型图: fig[1,1] 散点+拟合线, fig[1,2] 预测残差图,fig[2,1] residuals histrogram,fig[2,2]  residuals qqnorm
 
 ## Params
    1. data   df
    2. xlabel 预测变量  
    3. ylabel  响应变量
    4. model   回归模型
 ##  返回值
    fig  Makie 对象
"""
function plot_lm_res(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    y_hat=@pipe select(df,xlabel)|>predict(model,_)|>round.(_,digits=2)
    res=residuals(model)
    fig=Figure(resolution=(1000,800))
    Label(fig[0, 1:2], "$(xlabel)-$(ylabel)-Linear-Regression", fontsize = 24)

    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    ax3=Axis(fig[2,1],xlabel="rediduals",ylabel="frequency")
    ax4=Axis(fig[2,2],xlabel="quantiles",ylabel="residuals")
    
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,2];color = (:orange,0.1),strokewidth=0.3)
    scatter!(ax1,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    lines!(ax1,data[!,xlabel],y_hat,label="fit_line")
    stem!(ax2,res)
    hist!(ax3,res)
    qqnorm!(ax4,res;qqline = :fitrobust)
    return fig
end


"""
    plot_lm_res2(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    与 plot_lm_res 功能相同, y_hat 改为平方
    
`
xs=sort(data[!,xlabel])

y_hat=@pipe select(df,xlabel)|>predict(model,_)|>_.^2|>round.(_,digits=2)|>sort
`



TBW
"""
function plot_lm_res2(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    xs=sort(data[!,xlabel])
    y_hat=@pipe select(df,xlabel)|>predict(model,_)|>_.^2|>round.(_,digits=2)|>sort
    
    res=residuals(model)
    
    fig=Figure(resolution=(1000,800))
    Label(fig[0, 1:2], "$(xlabel)-$(ylabel)-Linear-Regression", fontsize = 24)

    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    ax3=Axis(fig[2,1],xlabel="rediduals",ylabel="frequency")
    ax4=Axis(fig[2,2],xlabel="quantiles",ylabel="residuals")
    
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,2];color = (:orange,0.1),strokewidth=0.3)
    scatter!(ax1,data[!,xlabel],data[!,ylabel].^2;marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    scatterlines!(ax1,xs,y_hat,label="fit_line")
    stem!(ax2,res)
    hist!(ax3,res)
    qqnorm!(ax4,res;qqline = :fitrobust)
    return fig
end


"""
    plot_reg_data(data::AbstractDataFrame, desc::Stat2Table, xtest::Vector{Float64}, yhat::Vector{Float64})

绘制一元回归曲线原始数据点和拟合曲线

## Arguments

1. data: 两列 dataframe , 预测变量为第一列, 响应变量为第二列
2. desc:  lock5stat table struct
3. xtest: 测试值
4. yhat: 预测值
"""
function plot_reg_data(data::AbstractDataFrame, desc::Stat2Table, xtest::Vector{Float64}, yhat::Vector{Float64})

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=desc.feature[1], ylabel=desc.feature[2], title=desc.question)
    scatter!(ax, eachcol(data)...; markersize=10, color=(:lightgreen, 0.3),
        strokecolor=:black, strokewidth=2)
    lines!(ax, xtest, yhat, label="fitting line", linewidth=3, color=:blue)
    axislegend()
    
    fig
end




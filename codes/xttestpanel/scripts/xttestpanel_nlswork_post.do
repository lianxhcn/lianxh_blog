* 推文第 3 节：nlswork 全样本诊断与函数形式子样本演示。
* 从项目根目录运行；args 可供外部启动器传入项目目录。
args project_root
if `"`project_root'"' != "" cd `"`project_root'"'
clear all
set more off
set linesize 100
version 14.0
capture mkdir logs
capture mkdir outputs
capture log close _all
log using "logs/xttestpanel_nlswork_post.log", text replace
capture confirm file "x/xttestpanel.ado"
if !_rc adopath ++ "x"
discard
capture which xttestpanel
if _rc {
    display as error "请先安装命令：ssc install xttestpanel"
    exit 199
}
which xttestpanel

* 全样本保留原始年份间隔，不将两次访谈强行当成相邻年份。
webuse nlswork, clear
xtset idcode year
xtreg ln_wage age tenure hours, fe
estimates store wage_fe
gen byte wage_sample = e(sample)
xttestpanel het
return list
estimates restore wage_fe
xttestpanel serial, lags(2)
return list
estimates restore wage_fe
xttestpanel csd
return list
estimates restore wage_fe
xttestpanel hausman
return list
estimates restore wage_fe
xttestpanel vif
return list
estimates restore wage_fe
xttestpanel func, reps(199)
return list

* 按个人随机抽取 300 人，保留被抽中者的全部访谈记录。
* 避免逐行抽样打碎个人的时间序列；子样本仅作教学演示。
preserve
keep if wage_sample
sort idcode year
set seed 20260919
by idcode: gen double draw = runiform() if _n == 1
by idcode: replace draw = draw[1]
egen byte tag = tag(idcode)
sort draw idcode year
gen long person_rank = sum(tag)
keep if person_rank <= 300
sort idcode year
assert _N <= 5000
xtreg ln_wage age tenure hours, fe
set seed 20260919
xttestpanel func, reps(199)
return list
restore
display "NLSWORK_POST_COMPLETE"
log close

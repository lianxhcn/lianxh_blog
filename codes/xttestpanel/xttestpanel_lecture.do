*======================================================================*
*  xttestpanel 中文讲义：线性面板模型的诊断测试
*  文件用途：课堂演示和学生自学用 Stata dofile
*  package_metadata_retrieved_date: 2026-06-17
*  从项目根目录运行本文件，例如：do xttestpanel_lecture.do
*======================================================================*

clear all
set more off
version 14.0

capture confirm file "x/xttestpanel.ado"
if !_rc adopath ++ "x"
capture which xttestpanel
if _rc {
    display as error "请先安装命令：ssc install xttestpanel"
    exit 199
}

capture mkdir "figs"
capture mkdir "figs/raw"
capture mkdir "logs"
capture mkdir "outputs"

capture log close _all
log using "logs/xttestpanel_lecture.log", replace text

*----------------------------------------------------------------------*
* 0. 环境检查
*----------------------------------------------------------------------*

* 优先使用项目内命令；否则使用读者通过 SSC 安装的版本。
discard

* 每次运行使用新的图片文件名，避免覆盖已上传或已引用的图片。
local run_date = string(daily("`c(current_date)'", "DMY"), ///
    "%tdCCYYNNDD")
local run_time = subinstr("`c(current_time)'", ":", "", .)
local figure_stamp "`run_date'-`run_time'"

which xttestpanel
help xttestpanel

display as text _newline "xttestpanel 是线性面板模型的诊断工具。"
display as text "它帮助我们检查误差结构、模型设定和解释变量问题。"
display as text "它不是自动修正工具；诊断后仍需要研究者判断。"

*----------------------------------------------------------------------*
* 1. 基本语法
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "基本语法"
display as text "{hline 70}"

display as text "推荐用法：先估计模型，再运行 postestimation 诊断。"
display as text "    xtset id year"
display as text "    xtreg y x1 x2 x3, fe"
display as text "    xttestpanel all"

display as text _newline "也可以使用 standalone 用法。"
display as text "    xttestpanel all y x1 x2 x3, model(fe)"

display as text _newline "常用选项包括："
display as text "    model(fe|re|tw|pool): 指定工作模型"
display as text "    graph: 输出单项诊断图"
display as text "    dashboard: 在 all 中输出组合诊断图"
display as text "    lags(#): serial 检验中的滞后阶数"
display as text "    reps(#): func 检验中的 wild bootstrap 次数"

*----------------------------------------------------------------------*
* 2. 教学模拟数据：把常见面板问题故意放进 DGP
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "教学模拟例子：数据中故意包含多种面板诊断问题"
display as text "{hline 70}"

set seed 20260609
local N = 50
local T = 20
set obs `=`N' * `T''

gen long id = ceil(_n / `T')
bysort id: gen int t = _n
xtset id t

* 个体效应 mu 与 x1 相关，因此 RE 的核心外生性条件不成立。
bysort id (t): gen double mu = rnormal() if _n == 1
bysort id (t): replace mu = mu[1]

gen double x1 = 0.6 * mu + rnormal()
gen double x2 = 0.4 * x1 + rnormal()
gen double x3 = 0.95 * x2 + 0.1 * rnormal()
gen double x4 = rnormal()

* 共同时间因子 ftime 会让不同个体的误差项相关。
bysort id (t): gen double load = 0.6 + abs(rnormal()) if _n == 1
bysort id (t): replace load = load[1]
bysort t (id): gen double ftime = rnormal() if _n == 1
bysort t (id): replace ftime = ftime[1]

* idiosyncratic error 同时包含 AR(1) 和随 x4 变化的方差。
sort id t
gen double e = .
replace e = rnormal() * sqrt(exp(0.4 * x4)) if t == 1
replace e = 0.5 * L.e + rnormal() * sqrt(exp(0.4 * x4)) if t > 1

gen double u = mu + load * ftime + e

* 真正的数据生成过程包含 x2^2，但估计模型故意遗漏该非线性项。
gen double y = 1 + 1.0 * x1 + 0.8 * x2 + 0.5 * x2^2 - 0.3 * x4 + u

summarize y x1 x2 x3 x4

*----------------------------------------------------------------------*
* 3. Standalone 用法：直接给 depvar 和 indepvars
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "Standalone 用法：xttestpanel all y x1 x2 x3 x4, model(fe)"
display as text "{hline 70}"

xttestpanel all y x1 x2 x3 x4, model(fe) dashboard reps(99)

scalar sim_p_het = r(p_het)
scalar sim_p_serial = r(p_serial)
scalar sim_p_csd = r(p_csd)
scalar sim_p_func = r(p_func)
scalar sim_p_hausman = r(p_hausman)
scalar sim_mean_vif = r(mean_vif)

tempname summary_handle
tempfile summary_data
postfile `summary_handle' str18 sample str60 data_source ///
    double(p_het p_serial p_csd p_func p_hausman mean_vif) ///
    using "`summary_data'", replace

post `summary_handle' ///
    ("simulation") ("Deliberately misspecified panel DGP") ///
    (sim_p_het) (sim_p_serial) (sim_p_csd) (sim_p_func) ///
    (sim_p_hausman) (sim_mean_vif)

capture graph export ///
    "figs/raw/xttestpanel-fig01-dashboard-`figure_stamp'.png", ///
    replace

display as text _newline "模拟数据的核心返回结果："
display as text "  r(p_het)     = " as result %8.4f sim_p_het
display as text "  r(p_serial)  = " as result %8.4f sim_p_serial
display as text "  r(p_csd)     = " as result %8.4f sim_p_csd
display as text "  r(p_func)    = " as result %8.4f sim_p_func
display as text "  r(p_hausman) = " as result %8.4f sim_p_hausman
display as text "  r(mean_vif)  = " as result %8.4f sim_mean_vif

display as text _newline "预期解释："
display as text "  异方差检验应拒绝，因为误差方差随 x4 改变。"
display as text "  序列相关检验应拒绝，因为 idiosyncratic error 包含 AR(1)。"
display as text "  截面相关检验应拒绝，因为所有个体受到共同时间因子影响。"
display as text "  函数形式检验应拒绝，因为线性模型遗漏 x2^2。"
display as text "  Hausman 检验应拒绝，因为 mu 与 x1 相关，RE 不一致。"
display as text "  VIF 会提示 x2 和 x3 存在严重共线性。"

*----------------------------------------------------------------------*
* 4. Postestimation 用法：先估计一次，再连续诊断
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "Postestimation 用法：先 xtreg，再 xttestpanel"
display as text "{hline 70}"

xtreg y x1 x2 x3 x4, fe

xttestpanel het, graph
capture graph export ///
    "figs/raw/xttestpanel-fig02-het-`figure_stamp'.png", replace

xttestpanel serial, lags(2) graph
capture graph export ///
    "figs/raw/xttestpanel-fig03-serial-`figure_stamp'.png", replace

xttestpanel csd, graph
capture graph export ///
    "figs/raw/xttestpanel-fig04-csd-`figure_stamp'.png", replace

xttestpanel vif, graph
capture graph export ///
    "figs/raw/xttestpanel-fig05-vif-`figure_stamp'.png", replace

xttestpanel hausman, graph
capture graph export ///
    "figs/raw/xttestpanel-fig06-hausman-`figure_stamp'.png", replace

xttestpanel func, reps(99)

*----------------------------------------------------------------------*
* 5. 示例数据：联网 nlswork 或本机 xtline1
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "示例数据：联网 nlswork 或本机 xtline1"
display as text "{hline 70}"

capture noisily webuse nlswork, clear
if _rc {
    display as error "webuse nlswork 失败，改用内置模拟纵向数据 xtline1。"

    sysuse xtline1, clear
    xtset person day
    bysort person (day): gen int trend = _n
    gen byte weekend = inlist(dow(day), 0, 6)
    gen double trend2 = trend^2

    local real_data "xtline1"
    local real_source "Stata built-in simulated longitudinal data"
    local real_dv "calories"
    local real_x "trend weekend trend2"
}
else {
    xtset idcode year

    local real_data "nlswork"
    local real_source "Stata webuse empirical labor panel"
    local real_dv "ln_wage"
    local real_x "age tenure hours"
}

xtreg `real_dv' `real_x', fe

xttestpanel all, reps(99)

scalar real_p_het = r(p_het)
scalar real_p_serial = r(p_serial)
scalar real_p_csd = r(p_csd)
scalar real_p_func = r(p_func)
scalar real_p_hausman = r(p_hausman)
scalar real_mean_vif = r(mean_vif)

post `summary_handle' ///
    ("`real_data'") ("`real_source'") ///
    (real_p_het) (real_p_serial) (real_p_csd) (real_p_func) ///
    (real_p_hausman) (real_mean_vif)
postclose `summary_handle'

preserve
use "`summary_data'", clear
export delimited using "outputs/xttestpanel-summary.csv", replace
restore

display as text _newline "`real_data' 的核心返回结果："
display as text "  r(p_het)     = " as result %8.4f real_p_het
display as text "  r(p_serial)  = " as result %8.4f real_p_serial
display as text "  r(p_csd)     = " as result %8.4f real_p_csd
display as text "  r(p_func)    = " as result %8.4f real_p_func
display as text "  r(p_hausman) = " as result %8.4f real_p_hausman
display as text "  r(mean_vif)  = " as result %8.4f real_mean_vif
display as text "诊断汇总表已保存到 outputs/xttestpanel-summary.csv。"

scalar list real_p_het real_p_serial real_p_csd real_p_func ///
    real_p_hausman real_mean_vif

*----------------------------------------------------------------------*
* 6. 如何阅读结果
*----------------------------------------------------------------------*

display as text _newline "{hline 70}"
display as text "结果阅读规则"
display as text "{hline 70}"

display as text "1. 先看原假设。p 值小，表示拒绝对应的理想条件。"
display as text "2. het 拒绝时，常规标准误可能不可靠。"
display as text "3. serial 拒绝时，需要考虑面板序列相关下的稳健推断。"
display as text "4. csd 拒绝时，需要考虑共同冲击、时间效应或 CCE 类方法。"
display as text "5. func 拒绝时，线性设定可能遗漏非线性或交互结构。"
display as text "6. Hausman 拒绝时，RE 一致性可疑，通常应优先考虑 FE。"
display as text "7. VIF 较高时，系数标准误可能被放大，但不等于因果识别失败。"

log close

* 连续 DID 系数分解：Stata 教学示例与本地核验
* 方法来源：Yu (2026), arXiv:2609.09488v2, Theorem 3。
* 本程序不是作者官方复现代码，也不是已发布的 Stata package。
* 本地核验详情见执行报告；已在 Windows、Stata 19.5 中运行。
* 结果已与 Python 的独立 OLS 和有理数基准交叉核验。
* 声明 version 17.0，本轮未实测 Stata 17；不依赖第三方命令。
* 注意：运行时会清空内存数据，并覆盖 results_stata 下的同名输出。
* 用法：将工作目录设为本文件所在文件夹，然后运行：
* do "continuous_did_stata_demo.do"

version 17.0
clear all
set more off
capture mkdir "results_stata"
capture log close cdid_v03
log using "results_stata/continuous_did_stata_demo.log", ///
    text replace name(cdid_v03)
display "开始本地 Stata 核验。以下 PASS 仅在本次运行通过断言后生成。"

* 每个城市保留一行；dy 是相同前后两期的结果变化量。
input byte id double dose double dy
1 0  0
2 0  0
3 1 10
4 3 10
end

* Step 1: 检查输入，并识别处理组和控制组。
assert !missing(id, dose, dy)
assert dose >= 0
isid id
generate byte treated = (dose > 0)
quietly count if treated == 0
assert r(N) > 0
quietly count if treated == 1
assert r(N) >= 2

* Step 2: 估计原连续 DID 系数。
quietly regress dy dose
scalar beta = _b[dose]

* Step 3: 计算处理组内的均值与方差。
* r(Var) 的分母为 n1-1；分解公式需要分母为 n1 的经验方差。
* 此处不要使用 meanonly，否则 summarize 不计算方差。
quietly summarize dose if treated == 1
scalar n1   = r(N)
scalar mu_d = r(mean)
scalar v_d  = r(Var) * (scalar(n1) - 1) / scalar(n1)
assert scalar(v_d) > 0 & scalar(v_d) < .

quietly count if treated == 0
scalar rho = r(N) / _N

* Step 4: 在处理组内部中心化处理强度，再估计两个组成部分。
* treated 的系数等于水平对比，z 的系数等于响应指数。
generate double z = treated * (dose - scalar(mu_d))
quietly regress dy treated z
scalar delta_l = _b[treated]
scalar theta_r = _b[z]

* Step 5: 计算权重、两个加权项，以及重构系数。
scalar lambda_r = scalar(v_d) / ///
    (scalar(v_d) + scalar(rho) * scalar(mu_d)^2)
scalar level_part = (1 - scalar(lambda_r)) * ///
    scalar(delta_l) / scalar(mu_d)
scalar response_part = scalar(lambda_r) * scalar(theta_r)
scalar beta_rebuilt = scalar(level_part) + scalar(response_part)

* 检查直接估计与分解重构是否一致。
assert abs(scalar(beta) - scalar(beta_rebuilt)) < 1e-10

* 只展示点估计，不使用四个观测进行显著性检验。
display "原连续 DID 系数 = " %9.4f scalar(beta)
display "水平对比        = " %9.4f scalar(delta_l)
display "组内回归系数    = " %9.4f scalar(theta_r)
display "组内系数权重    = " %9.4f scalar(lambda_r)
display "分解重构系数    = " %9.4f scalar(beta_rebuilt)

* 四种情形的统一核验。
* case 1: 处理效应不随正强度变化。
* case 2: 处理效应与强度成正比。
* case 3: 组间差异与组内系数异号。
* case 4: 增加控制组，保持处理组不变。
matrix Results = J(4, 7, .)
matrix Expected = (10/3, 10, 0, 1/3, 10/3, 0, 10/3 \ ///
    2, 4, 2, 1/3, 4/3, 2/3, 2 \ ///
    2, 10, -4, 1/3, 10/3, -4/3, 2 \ ///
    15/4, 10, 0, 1/4, 15/4, 0, 15/4)

forvalues k = 1/4 {
    clear
    if `k' == 4 {
        quietly set obs 8
    }
    else {
        quietly set obs 4
    }
    generate double dose = 0
    quietly replace dose = 1 in 1
    quietly replace dose = 3 in 2
    generate double dy = 0
    if inlist(`k', 1, 4) {
        quietly replace dy = 10 in 1/2
    }
    if `k' == 2 {
        quietly replace dy = 2 * dose
    }
    if `k' == 3 {
        quietly replace dy = 14 in 1
        quietly replace dy = 6 in 2
    }
    generate byte treated = (dose > 0)

    quietly regress dy dose
    scalar beta = _b[dose]
    quietly summarize dose if treated == 1
    scalar n1 = r(N)
    scalar mu_d = r(mean)
    scalar v_d = r(Var) * (scalar(n1) - 1) / scalar(n1)
    quietly count if treated == 0
    scalar rho = r(N) / _N
    generate double z = treated * (dose - scalar(mu_d))
    quietly regress dy treated z
    scalar delta_l = _b[treated]
    scalar theta_r = _b[z]
    scalar lambda_r = scalar(v_d) / ///
        (scalar(v_d) + scalar(rho) * scalar(mu_d)^2)
    scalar level_part = (1 - scalar(lambda_r)) * ///
        scalar(delta_l) / scalar(mu_d)
    scalar response_part = scalar(lambda_r) * scalar(theta_r)
    scalar beta_rebuilt = scalar(level_part) + scalar(response_part)
    assert abs(scalar(beta) - scalar(beta_rebuilt)) < 1e-10

    * 独立核对：水平对比应等于两组均值差。
    quietly summarize dy if treated == 1, meanonly
    scalar treated_mean = r(mean)
    quietly summarize dy if treated == 0, meanonly
    scalar control_mean = r(mean)
    assert abs(scalar(delta_l) - ///
        (scalar(treated_mean) - scalar(control_mean))) < 1e-10

    * 独立核对：组内系数应等于处理组内协方差与经验方差之比。
    generate double cov_term = ///
        (dose - scalar(mu_d)) * (dy - scalar(treated_mean)) ///
        if treated == 1
    quietly summarize cov_term if treated == 1, meanonly
    scalar theta_moment = r(mean) / scalar(v_d)
    assert abs(scalar(theta_r) - scalar(theta_moment)) < 1e-10

    matrix Results[`k', 1] = (scalar(beta), scalar(delta_l), ///
        scalar(theta_r), scalar(lambda_r), scalar(level_part), ///
        scalar(response_part), scalar(beta_rebuilt))
    forvalues j = 1/7 {
        assert abs(el(Results, `k', `j') - ///
            el(Expected, `k', `j')) < 1e-10
    }
    display "case `k': PASS"
}

matrix colnames Results = beta delta_L theta_R lambda_R ///
    level_component response_component beta_rebuilt
matrix rownames Results = fixed linear opposite more_controls
matrix list Results, format(%10.4f)

* 导出真正由本地 Stata 生成的结果，不与 Python 基准文件混名。
clear
svmat double Results, names(col)
generate byte case_id = _n
order case_id
export delimited using "results_stata/stata_results.csv", replace

display "STATA_CHECK: PASS (4 cases; all assertions passed)"
log close cdid_v03

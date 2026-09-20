* 本文件与推文中的 Stata 核心代码一致。
* 本地核验详情见执行报告；运行前请先保存内存中的数据。
version 17.0
clear all
set more off

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

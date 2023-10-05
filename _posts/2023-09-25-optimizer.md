---
title: "옵티마이저(Optimizer)"
date: 2023-09-25 00:00:00 +09:00
categories: DeepL
published: true
math: true
---

# Optimizer

## 경사하강법

---

$$
W_{t+1}=W_t-\eta \cdot \triangledown L(W_t)
$$

- 최적화 이론중 $\hat{\theta}^{MLE}$의 명확한 solution을 구할 수 없을 때 뉴튼-랩슨 반복법으로 해를 구한다.
- 바틀렛 항등식과 테일러 급수의 근사를 이용하여 유도한 방식으로, $\dot{L}(W_{t+1})\approx \dot{L}(W_t)+\ddot{L}(W_t)\cdot (W_{t+1}-W_t)$ 이를 정리하면 아래와 같은 반복법을 유도할 수 있다.
  - ${W}_{t+1}={W}_t - {\dot{L}({W}_t) \over \ddot{L}({W}_t)}$
- 1차(Jacobian), 2차(Hessian) 편도함수가 모두 필요하다.
- Hessian matrix 정보까지 활용하므로 1차 근사보다 최적점으로 수렴이 빠르다는 장점이있다.
- 하지만, Hessian matrix를 구하려면 계산비용이 많이 드는데 고전 통계학에서는 이런 큰 다차원 파라미터를 만질일이 잘 없었지만 ML/DL 분야에서는 파라미터 수가 커도 너무 크므로 계산비용도 덩달아 커진다.
- 따라서, 정확한 Hessian 대신 학습률을 사용하는 방법론이 머신러닝, 딥러닝에서 흔히 쓰이는 경사하강법이다.

## 확률적 경사하강법

---

![Untitled](https://blog.kakaocdn.net/dn/bpTLYE/btqBn8JJDd0/VTRXK6s5klIghDhuW9qH50/img.png)

$$
W_{t+1}=W_t-\eta \cdot \triangledown L(W_t\;;x^{(batch)},y^{(batch)})
$$

- 신중하게 한걸음 가는 것은 너무 느리다. 경사에 몸을 맡겨서 내려가자.
- mini-batch이므로 full-batch일 때의 gradient와 정확히 일치하진 않지만 얼추 비슷하게 내려간다(비틀비틀 내려가는 이유).
- 1걸음 갈 시간에 5걸음씩 가므로 계산비용이 대폭 줄어들고 엉뚱하게 수렴하지도 않으므로 현재 거의 모든 학습은 배치단위로 이루어진다.
- 단, 학습률이 0으로 수렴하지않으면 최적점으로 수렴하지 못한다는 단점이있다.
- 단순 SGD방식으론 Local minimum에서도 빠져나오기 힘들다는 단점도 있다.

## 모멘텀

---

$$
W_{t+1}=W_t-m_t
$$

$$
m_t=\beta \cdot m_{t-1}+ (1-\beta)\cdot \triangledown L(W_t)
$$

- $m_t$를 관성(모멘텀)벡터라 하는데 물리학에서 영감을 얻은 아이디어이다.
- $m_t$를 풀어쓰면 다음과 같다.

$$
W_{t+1}=W_t-\eta \cdot \triangledown L(W_t)- \beta \cdot m_{t-1}
$$

- 기본적인 경사하강법처럼 $-\eta \cdot \triangledown L(W_t)$로 한발 내려간다음 추가적으로 한발을 더 내딛을 건데, $-\beta \cdot m_{t-1}$ 만큼 더 가겠다는 의미이다.
- $m_{t-1}$은 이전시점($t-1$)에서 가중치 변화량을 의미한다.
- 이전 시점의 방향과 현재시점의 방향이 같으면 더욱 가속도를 얻고 다르면 저항을 받는다고 볼 수 있다.
- Local minimum은 보다 더 잘 탈출하지만 여전히 수렴하기까지 u-turn이 많다.

## NAG

---

$$
W_{t+1}=W_t-\beta \cdot \Delta W_{t-1}-\eta \cdot \triangledown L(W_t-\beta \cdot \Delta W_{t-1})
$$

- 모멘텀 방식에서 정확히 순서만 바꾼 방법이다.

  ![Untitled](https://blog.kakaocdn.net/dn/OnxCe/btrbOQGqlth/CaAzY3T4RwP7x9bFWghkek/img.png)

- 즉, 먼저 관성이동을 하고 그 위치에서 일반적인 경사하강 이동을 하겠다는 의미이다.

## Adagrad

---

$$
W_{t+1}=W_t-\eta \cdot {g_t\over \sqrt{v_t+\epsilon}}
$$

$$
v_t=v_{t-1}+ g^2_t
$$

- 타임스텝 $t$가 증가할수록 누적 gradient 크기 기반으로 학습률을 줄이겠다는 의미이다.
- 따라서, 학습이 진행될 수록 최적점 주변을 맴돌던 현상을 일명 담금질 기법으로 수렴시키겠다는 아이디어이다.
- convex 함수에서는 아주 좋지만 non-convex 함수에서는 초기 그레디언트가 크면 학습률이 너무 빠르게 식기 때문에 Local minimum을 탈출하기 전에 갇히는 단점이있다.
- 같은 맥락으로 학습률의 선택에도 굉장히 민감하다는 단점이있다.

## RMSProp

---

$$
W_{t+1}=W_t-\eta{g_t\over \sqrt{E[g^2]_t+\epsilon}}
$$

$$
E[g^2]_t=\beta \cdot E[g^2]_{t-1}+(1-\beta)\cdot  g^2_t
$$

- Adagrad의 문제점을 해결하기위해 $v_t$에 지수이동평균을 방식을 적용한 형태이다.
- 이렇게 해줌으로써 현재 그레디언트 크기의 영향력을 ($1-\beta$)만큼 penalty를 줌과 동시에 과거 누적도 계속해서 $\beta$만큼 penalty를 먹이는 형태가 되어 Adagrad보다 **천천히** 학습률이 감소한다.

## Adadelta

---

- RMSProp과 비슷한 시기에 독립적으로 연구된 최적화 방법으로 RMSProp과 많이 유사하지만 차이점은 있다.
- 우선, Adagrad의 문제점이 2가지가 있었다.
  1. $t$가 증가할수록 $\eta$가 빠르게 작아지는 문제
  2. $\eta$ 초기 설정에 민감한 문제
- 1번 문제를 해결하기위해 $g^2_t$ 대신 평균을 씌워 $E[g^2]_t$ 로 대체하였고 윈도우 누적을 사용하여 과거 시점의 모든 $g$를 사용하는 것이 아닌 고정된 윈도우(일정 범위$g$)로 제한했다.
  - $$E[g^2]_t=\beta \cdot E[g^2]_{t-1}+(1-\beta)\cdot g^2_t$$

수식은 다음과 같다.

$$
\begin{align} W_{t+1}&=  W_t- \eta \cdot {g_t \over \sqrt{E[g^2]_t + \epsilon}} \\&=W_t- \eta \cdot {g_t \over RMS[g]_t}
\end{align}
$$

여기에서 논문의 저자는 가중치와 업데이트 가중치 간의 **단위 차이**가 난다는 것을 지적하면서 가중치 업데이트의 평균 제곱근 오차 $RMS[\Delta W]_t$ 를 사용하여 학습률$\eta$를 대체한다.

- $$RMS[\Delta W]_t=\sqrt{E[\Delta W^2]_t+\epsilon} \qquad ,E[\Delta W^2]_t=\beta \cdot E[\Delta W^2]_{t-1}+(1-\beta)\cdot \Delta W^2_t$$

$$RMS[\Delta W]_t$$ 은 Unknown이므로 $RMS[\Delta W]_{t-1}$ 로 근사하여 최종적인 수식은 다음과 같다.

$$
 W_{t+1}=W_t- {RMS[\Delta W]_{t-1} \over RMS[g]_t} \cdot g_t
$$

- 즉, 학습률 $\eta$를 제거하면서 2번 문제를 해결하고자 한 것이다.

## Adam

---

$$
W_{t+1}=W_t-\eta {\frac{m_t}{\sqrt{v_t+\epsilon}}}
$$

- 위에서 설명했던 몇몇 아이디어를 흡수한 하이브리드 버전이다 (**Momentum** + **Adaptive**).
- $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ (**Momentum**)
- $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ (**Adaptive**)

위와같이 이동평균방식으로 정의한 이유는 시점t에서의 단일추정치인 $g_t$에만 의존하고 싶지 않았기 때문이었다. 한편, 우리는 SGD방식을 사용한다는 점을 고려하면 우리가 관심있는 것은 $g_t$ 보다 $E[g_t]$ 일 것이다. 우리는 $E[g_t]$의 추정량으로서 $m_t$를 계산하고자 하며 이상적인 경우 다음과 같이 되길 원한다.

$$
E[m_t]=E[g_t]
$$

위 식을 만족할 경우 $m_t$를 $E[g_t]$에 대한 불편추정치라 부른다.

자, 이제 반복법으로 $m_t$와 $v_t$를 구하기 위해 다음과 같이 초기화를 할 것이다. $m_0=0,v_0=0$

$m_t$와 $v_t$ 모두 동일한 과정을 거치므로 $m_t$에 대한 예시만 들어보겠다.

$$
\begin{align*}
m_0 = &0 \\
m_1=&0.9m_0 + 0.1 g_1 = 0.1g_1 \\
m_2 = & 0.9m_1+0.1g_2=0.09g_1+0.1g_2\\
\vdots \\
m_t= & (1-\beta_1)\sum^t_{i=1}\beta_1^{t-i}g_i
\end{align*}
$$

위와 같은 식을 얻을 수 있다.

이제 우리가 원하는 이론을 충족하기위해 양변에 기댓값을 취하면 다음과 같다.

$$
\begin{align*}
E[m_t] &= E[(1-\beta_1)\sum^t_{i=1}\beta_1^{t-i}g_i]\\
&= (1-\beta_1)E[\sum^t_{i=1}\beta_1^{t-i}g_i] \\
&= (1-\beta_1)\sum^t_{i=1}E[\beta_1^{t-i}g_i] \\
&= (1-\beta_1)\sum^t_{i=1}\beta_1^{t-i}E[g_i]
\end{align*}
$$

여기서, 모든 $g_i$들이 같은 분포에서 나왔다고 가정하면(즉, $iid$를 가정하면) $E[g_i]=E[g_t]$이므로

$$
\begin{align*}

&= (1-\beta_1)\sum^t_{i=1}\beta_1^{t-i}E[g_t] \\
&= (1-\beta^t)E[g_t]
\end{align*}
$$

즉, 정리하면 $E[{m_t\over 1-\beta^t}]=E[g_t]$ 이고 $m_t$ 대신 $\hat{m}_t={m_t\over 1-\beta^t}$ 가 편향을 보정한 추정치가 되는 것이다.

직관적으로는 $t\rightarrow \infty$ 일때 $(1-\beta^t) \rightarrow 1$ 로 수렴하므로 일정 시점이 지나면 보정효과가 사라져감을 알 수 있다. 이것은 초반에 초기값($i.e. \;0$)으로 편향되는 것을 보정하기 위함이다.

$$
W_{t+1}=W_t-\eta {\hat{m}_t\over \sqrt{\hat{v}_t+\epsilon}}
$$

## AdamW

---

일반적으로 정규화 기법중 $L_2-norm$과 $weight\;decay$ 는 동일한 작용을 한다는 믿음에도 불구하고 SGD에서는 동일하지만 Adam과 같은 Adaptive 방법에서는 동일하지 않다는 것을 증명했다.

우선 가장 기본적인 Weight decay는 다음과 같다.

$$
W_{t+1}=(1-\lambda)W_t-\eta \cdot g_t
$$

$L_2$ 정규화는 다음과 같이 표현된다.

$$
W_{t+1}=W_t-\eta \cdot {\delta\over \delta W_t}(L(W_t)+{\lambda^{'}\over 2}\vert\vert W_t\vert\vert^2_2)
$$

SGD방식에서는 두개의 식이 $\lambda^{'}={\lambda \over \eta}$로 주어지는 경우 정확히 일치함을 쉽게 보일 수 있다.

여기서 알 수 있는점은 $\lambda^{'}$는 학습률$\eta$에 의존적이기 때문에 최적의 $\lambda^{'}$를 찾았더라도 $\eta$가 변함으로써 최적이 아니게 될 수 있음을 시사한다. 역사적으로 SGD는 이러한 정규화 방식을 계승했으며 Adam도 마찬가지였다.

하지만 Adam에서 $L_2$ 정규화와 Weight decay가 동일하지 않음을 밝히면서 ($L_2$에서는 gradient가 큰 가중치는 상대적으로 적은 양으로 정규화됨과 대조적으로 Weight decay는 동일한 비율로 정규화됨) Adam에서 Weight decay를 사용하려면 다음과 같이 decouple 하여 업데이트 룰을 수정해야 한다(decoupled weight decay).

![Untitled](https://ml-explained.com/articles/adamw-explained/adamw.png)

ref.

[https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
[https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)

# Physics-informed neural networks (PINNs)

Task:

<img src= "./task.png"  alt= "Description of Task"/>

PINNs original paper [2019] can be accessed <a href="https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999118X00229/1-s2.0-S0021999118307125/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFZHBTOGp0kSQDLGh2nHzY6in8p8jJ1j%2BbVgnCLB7SBAAiAfx6zWodeBX00zE53OPePcLlLBKL4ZAtZnXLPIt8pFUSqyBQhpEAUaDDA1OTAwMzU0Njg2NSIMR1T5ZXlTfOVLZ6lKKo8Fe9Eh2G3HU7JyrLtBT0zVduSXrur9G5dUg6jhKlmVb0dzzYz%2Fu6lK%2F0HpYdclC0vP2QxjtUiR1F0whQ93f95ioycgdTcorR5NAe9tdnYcUExiKIIu4wjcQshoJs45NXMy6JucKqzqJhIMtFKyHSHUb0XUSK6cRlxngy5OkDb5vgWHMnQFc1rD5mYXYeBtO7y84oQdbZ6j5s9Xt7RlJuvd3MTUplka4IJkHasUTMJvb0UBkIxyzuTJW6EBve2NFrMkQEggQ%2Brt2l%2Fi%2FFiY0HQw8cmIozS6wuGOBvF8Tv%2BtZYdRcfYpNfMhyNpFqY3Ceozg99qJJcHKOjTyA0MKzzlS6MEM1Pi%2FOOkmw9lc4kJR8nSR0mdC4sK7IIJGp8MwRnu%2B30z8QjqGGtvgDaJmCOfo4vsFsPdi7yQwqgbmd%2BN%2B0aR8tVaPWV72dhggXIppJaZhIoRS3hnRj59DGHADKFyQhASRxtjIOn3Az4TmmvA%2Fzk%2BGw2s4nYmcgCKfsjiwbQ81%2FTYKTn0DfD5uDpdtn3cOxLrqTDz0BvBgi0srrJZKUVfhAPwcyeFS4Guxdq8AluMeKb43tS4DVE0k0WgjjmFVEnz5NTNYSOXNVzTkn4ukWsqYYuO3K1yaQ2NuJW6%2BZiI5dwCTNGe3aSt6aGeoSVobbLNNdUJomAnHp4K%2BuRs7VzVw%2B5JXH13MyMgSuQDXN3YVe%2Ba54uoXHItE9XkIp12qgi%2FQQPNd6e%2Ff3VxewYTVuvu6aHTXGG92m52QhLK4sClgd3xD3ZOc08qBIOmtZSmrNoWM5vh2sBgETx5jF5FvJSR%2BcxrSDZ62N1IjFqVJVYiqTbA9UHou7VG7XwdbT777UpSu48oeBewz354k2t5R4jCM1KquBjqyARL6mP2AYb%2B4MA7yGkC2dWJmK%2BWl0hXdJDsSchvNDIDdTMSkAs9pjsJg3KGTlj4CvOADr0eKKJT1Iv5Hitpo8OS3dVXCE13ZyAUL9zPWYVxvZRFBoitiwBcXXmengj30%2BmTf2nHjBoPxxNTLeU358uKOE9YYZr8acaO3N7Xr5lxCNliHKHltvrgAC0tkt9jJPs5PkxyBWd%2Bwq%2FZtiVCPVkL9gC2cimtSPKzWrc6x9omNwU8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240213T002503Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVSYWBV7C%2F20240213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1c48251ec31afdc4f0b1dad3685b0c20cead013d1937bcdaa62e61621615bfd5&hash=432c36f9b594bde8de4c54dafc5ec9edb3fe3921f0238e860a2deb3f72052792&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021999118307125&tid=spdf-33f995d5-1e73-4a3d-91ca-f7bab0ea642a&sid=e9952ef7384d024d51388b998269855ef852gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=15115c5b5558030259&rr=8548ee97788942e9&cc=us"> here. </a>

The remaining of this document is structured as follows:

1. Equations: Navier-Stokes and Schrodinger
2. PINNs : How they work and which gap they fill in the literature.
3. PINNs : Limitations.
4. Reproducing paper experiments: Pinns-torch
5. Discussion and Conclusion


## 1. Equations

##### Navier-Stokes

Some people love it, other people hate it. However, it is undeniable that solving Navier-Stokes and fully understanding it remains a million dollar question (or should I say equation? 😄).

<p align="center"> <img src="./navier_fun.png" height="400px" width="400px" alt="Image from a youtuber's love for navier stokes equation."/> </p>


In Raissi et al., they experiment PINNs on the 2D version of Navier Stokes equation. The equation is described explicitly  as follows:

$$
u_t + \lambda_{1}(uu_x + vu_y) = -p_x + \lambda_{2}(u_{xx} + u_{yy}), 

$$

$$
v_t + \lambda_{1}(uv_x + vv_y) = -p_y + \lambda_{2}(v_{xx} + v_{yy})

$$

where $u(t,x,y)$ denotes the x-component of the velocity field, $v(t,x,y)$ the y-component, whereas the pressure is described by $p(t,x,y)$.  I highlight that in this setup, we have no knowledge about the $\lambda$ values ($\lambda_1 , \lambda_2$).

For the purposes of the paper and this assignment, the experiments consider the pseudo problem of a  "incompressible flow past a circular cylinder" (Raissi et al., 693) using the Navier Stokes 2D formulation shown above.

##### Schrodinger

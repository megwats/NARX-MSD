%% 1. Load data (assumes same txt layout as Python) [t; u; x; v]
data = dlmread('N=10000-dt=2.5e-05-force=sin(2pi200t).txt', '\t');
N      = 10000;
t      = data(1:N);
u      = data(N+1:2*N);
x      = data(2*N+1:3*N);
v      = data(3*N+1:4*N);

%% 2. z-score normalisation (per signal)
[xn, x_mu, x_sig] = zscore_vec(x);
[un, u_mu, u_sig] = zscore_vec(u);
[vn, v_mu, v_sig] = zscore_vec(v);

%% 3. Continuous-time MSD parameters + Tustin discretisation
m    = 1;
f_n  = 100.0;
zeta = 0.015;
wn   = 2*pi*f_n;
k    = (2*pi*100)^2;
c    = 2*zeta*m*wn;

dt   = 2.5e-5;
I2   = eye(2);
A2   = [0 1; -k/m -c/m];
B2   = [0; 1/m];

Adisc = (I2 - 0.5*dt*A2)\(I2 + 0.5*dt*A2);
Bdisc = (I2 - 0.5*dt*A2)\(dt*B2);

%% 4. Build Z = [x_{k-1}, v_{k-1}, u_{k-1}], Y = [x_k, v_k]
Z = [x(1:end-1), v(1:end-1), u(1:end-1)];   % (N-1)×3
Y = [x(2:end),   v(2:end)  ];               % (N-1)×2

%% 5. Closed-form LS initialisation of W_star (3×2)
W_star = (Z.'*Z)\(Z.'*Y);                   % (3×3)\(3×2) -> 3×2

C_tustin  = [Adisc Bdisc];                  % 2×3
C_ls      = W_star.';                       % 2×3
disp('Tustin-based weights:');  disp(C_tustin);
disp('LS-based weights:');      disp(C_ls);@

%% 6. NARX parameters and training settings
W      = C_tustin;     % or C_ls
lr     = 1e-3;
epochs = 5e5;
tol    = 1e-9;

Ztr = [xn(1:end-1), vn(1:end-1), un(1:end-1)];  % (N-1)×3
Ytr = [xn(2:end),   vn(2:end)  ];               % (N-1)×2

%% 7. Batch training loop (MSE on 1-step prediction)
for epoch = 0:epochs
    Pred = Ztr * W.';                % (N-1×3)*(3×2) -> (N-1×2)
    E    = Pred - Ytr;
    loss = mean(E(:).^2);

    M  = size(Ztr,1);
    dW = (2/M) * (E.' * Ztr);        % (2×(N-1))*(N-1×3) -> 2×3

    W = W - lr * dW;

    if mod(epoch, 1e5) == 0
        fprintf('Epoch %d, Loss %.3e\n', epoch, loss);
    end
    if loss < tol
        fprintf('Early stopping at epoch %d, Loss %.3e\n', epoch, loss);
        break;
    end
end

%% 8. Predict on full trajectory (normalised) and de-normalise x-component
xv_pred_n = narx_forward(un, W, [xn(1); vn(1)]);  % 2×N
x_pred_n  = xv_pred_n(1,:).';                     % N×1

x_pred = x_pred_n * x_sig + x_mu;

idx = 1:4000;
figure; hold on;
plot(idx, x_pred(idx), 'r', 'DisplayName', 'NARX x');
plot(idx, x(idx),      'k--', 'DisplayName', 'True x');
xlabel('k'); ylabel('x');
legend; grid on;

mse_x = mean((x_pred(:) - x(:)).^2);
fprintf('MSE on x: %.3e\n', mse_x);

%% ===== Local functions (must be at end) =====

function x_out = narx_forward(u_in, W, x0)
    % u_in: N×1, W: 2×3, x0: 2×1 (all normalised)
    N     = numel(u_in);
    x_out = zeros(2, N);
    x_out(:,1) = x0;
    for k = 2:N
        h_prev = x_out(:,k-1);        % 2×1
        u_prev = u_in(k-1);           % scalar
        z      = [h_prev; u_prev];    % 3×1
        x_out(:,k) = W * z;           % 2×1
    end
end

function [x_norm, m, s] = zscore_vec(x)
    x = x(:);
    m = mean(x);
    s = std(x, 1);      % population std
    if s == 0
        s = 1;
    end
    x_norm = (x - m)/s;
end

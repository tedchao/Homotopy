function driv
% Homotopy continuation for power flow for s=1 connected component

% this version uses lsqr for linear systems
% and can use various preconditioners

% matrix A dimensions is n by n
n = 16;

% construct sparse A0 and A1 with random values
rng(0);
running_example = @directed_cycle_matrix;
running_example = @cycle_matrix;
running_example = @ladder_matrix;

A0 = running_example(n);
ep = 0.05;
A1 = A0 + ep*running_example(n);

% choose a solution
x0 = my_random_matrix(n,1);
y0 = my_random_matrix(n,1);
x0(n) = 0;
y0(n) = 0;

zero_b = zeros(2*n,1);

% right-hand side for this solution
b0 = F(x0, y0, A0, A1, zero_b, zero_b, 0);

% choose a solution
x1 = x0 + ep*my_random_matrix(n,1);
y1 = y0 + ep*my_random_matrix(n,1);
x1(n) = 0;
y1(n) = 0;

% right-hand side for this solution
b1 = F(x1, y1, A0, A1, zero_b, zero_b, 1);

x = x0; y = y0;

% check rank
% J = Jac(x0, y0, A0, A1, 0.1);
% svd(J)
%[q,r] = imgs(J);
%size(r)
%spy(r)

numsteps = 10;
deltat = 1/numsteps;
t = 0;

for step = 1:numsteps
  % predictor using tangent
  J = Jac(x, y, A0, A1, t);
  f = dFdt(x, y, A0, A1);
  inc = -J \ f;
  x(1:n-1) = x(1:n-1) + deltat*inc(1:n-1);
  y(1:n-1) = y(1:n-1) + deltat*inc(n:end);
  t = t + deltat;
  
  fprintf('======== time step %d: %f =========\n', step, t);
  fprintf('After predict: %f\n', norm(F(x, y, A0, A1, b0, b1, t)));

  % corrector using Gauss-Newton iterations
  for iter = 1:10
    J = Jac(x, y, A0, A1, t);
    J = sparse(J); % UNDONE: form J directly as sparse matrix
    f = F(x, y, A0, A1, b0, b1, t);
    % inc = -J(:,1:end-1) \ f;
%   jtj = J(:,1:end-1)'*J(:,1:end-1); R = ichol(jtj)'; % ichol pivot fails
%   jtj = J(:,1:end-1)'*J(:,1:end-1); R = chol(jtj);
%   [~, R] = qr(J(:,1:end-1)); R=R(1:end-1,:);
    [~, R] = imgs(J);

%   R = precon_sgs(J(:,1:end-1));
%   R = eye(2*n-1); % no preconditioning
%   R = precon_sgs(J(:,1:end-1)); % also init guess for paric_ref
%   jtj = J(:,1:end-1)'*J(:,1:end-1); jtj=sparse(jtj); [R dummy] = paric_ref(jtj, R, 1); % pivot fails
    [inc flag relres lsqr_iter] = lsqr(J, -f, 1e-6, 1000, R);
    assert(flag == 0, 'LSQR failed');
    fprintf('     LSQR steps %d\n', lsqr_iter);
    x(1:n-1) = x(1:n-1) + inc(1:n-1);
    y(1:n-1) = y(1:n-1) + inc(n:end);
    res = norm(F(x, y, A0, A1, b0, b1, t));
    fprintf('Newton step %d: %f\n', iter, res);
  end
  norm_inc = norm(inc); 
  if norm_inc > 0.0001 
     fprintf('|inc| is HUGE!!!\n');
     svd(J)
     return;
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F(x, y, A0, A1, b0, b1, t)
C = (1-t)*A0 + t*A1;
b = (1-t)*b0 + t*b1;
n = length(C);
z = x + i*y;
f = diag(z)*(C*conj(z)) - (b(1:n)+i*b(n+1:end));
f = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = dFdt(x, y, A0, A1);
T = (A1-A0);
z = x + i*y;
f = diag(z)*(T*conj(z));
f = [real(f); imag(f)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = Jac(x, y, A0, A1, t);
C = (1-t)*A0 + t*A1;
n = length(C);
Cx = C*x;
Cy = C*y;
J = [diag(x)*C+diag(Cx)    diag(y)*C+diag(Cy)
     diag(y)*C-diag(Cy)   -diag(x)*C+diag(Cx)];
J = [J(:,1:n-1) J(:,n+1:end-1)];
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r = my_random_number();
r = rand * 2 - 1;
function r = my_random_matrix(m,n);
r = rand(m,n) * 2 - 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = directed_cycle_matrix(n);
A = spdiags(my_random_matrix(n,2),0:1,n,n);
A(n,1) = my_random_number;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = cycle_matrix(n);
A = spdiags(my_random_matrix(n,3),-1:1,n,n);
A(n,1) = my_random_number;
A(1,n) = my_random_number;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = ladder_matrix(n);
assert(mod(n,2)==0, 'even n expected');
A = spdiags(my_random_matrix(n,3),-1:1,n,n);
for i=1:n
  j = n+1-i;
  A(i,j) = my_random_number;
  A(j,i) = my_random_number;
end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = precon_sgs(J)
% upper triangular Symmetric Gauss-Seidel preconditioner
jtj = J'*J;
d = 1 ./ sqrt(diag(jtj));
R = diag(d) * triu(jtj);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a r] = imgs(a)
% incomplete modified Gram-Schmidt
% r is a square upper triangular matrix
% after factorization: input_a'*input_a = r'*r
% factorization does not break down unless exact zero pivot encountered

% note that the diagonal of r is not scaled to be positive
% resulting a has orthogonal columns (not normalized)

n = size(a,2);
pat = spones(a'*a); % choose pattern of r factor
r = zeros(n,n);

for i = 1:n
  r(i,i) = norm(a(:,i));
  q(:,i) = a(:,i)/r(i,i);
  for j = i+1:n
    if pat(i,j) ~= 0
      r(i,j) = q(:,i)'*a(:,j);
      a(:,j) = a(:,j) - r(i,j)*q(:,i);
    end
  end
end
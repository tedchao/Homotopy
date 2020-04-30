function J = homotopy(A1,pertub,numsteps,seed)

rng(seed);

n = length(A1);

% Construct A0
A0 = (1-pertub*rand(1))*A1;

% Solution for start system
[x0,y0] = getsolution(n);
x0(n) = 1; 
y0(n) = 0;

% Right-hand side of start system
zero_b = zeros(2*n,1);
b0 = F(x0, y0, A0, A1, zero_b, zero_b, 0);

%check rank of Jacobian
J = Jacobian(x0, y0, A0, A1, 0);
%cond(J)
%size(J)
%nnz(J)
%rank(J)

%[~,R] = imgs(J);
%cond(J*inv(R))
%setup.type = 'crout'; 
%[l,u] = ilu(sparse(J),struct('type','ilutp','milu','col','droptol',1e-3));

%cond(J*inv(u))

% Solution for target system
x1 = x0 + pertub*rand(n,1);
y1 = y0 + pertub*rand(n,1);
x1(n) = 1; 
y1(n) = 0;

% Right-hand side of target system
b1 = F(x1, y1, A0, A1, zero_b, zero_b, 1);

x = x0; y = y0;
delta = 1/numsteps;
t = 0;
gmres_step = 0;
total_step = 0;

%[l,u] = ilu(sparse(J),struct('type','crout','droptol',1e-3));
%[l,u] = ilu(sparse(J),struct('type','ilutp','milu','col','droptol',1e-3));
r = zeros(numsteps,1);
for step = 1:numsteps
    %predictor
    J = Jacobian(x, y, A0, A1, t);
    f = dFdt(x, y, A0, A1);
    dxdt = -J\f;    %size is 198x1, when n=100
    x(1:n-1) = x(1:n-1) + delta*dxdt(1:n-1);
    y(1:n-1) = y(1:n-1) + delta*dxdt(n:end);

    t = t + delta;
    fprintf('======== time step %d: %f =========\n', step, t);
    fprintf('After predict: %f\n', norm(F(x, y, A0, A1, b0, b1, t)));
    
    % reuse the preconditioner
    %J = Jacobian(x, y, A0, A1, t);
    %[l,u] = ilu(sparse(J),struct('type','ilutp','milu','col','droptol',1e-3));
    %[l,u] = ilu(sparse(J),struct('type','crout','droptol',1e-3)); %iLU works
    
    %corrector using Newton
    for iter = 1:10
        J = Jacobian(x, y, A0, A1, t);
        f = F(x, y, A0, A1, b0, b1, t); f = [f(1:n-1); f(n+1:end-1)];
        
        %{%
        if iter == 1
            [l,u] = ilu(sparse(J),struct('type','crout','droptol',1e-3)); %iLU works
            %[s,flag,relres,gmres_iter] = gmres(J, -f, [], 1e-6, 40,l,u);
        else
            B = J - J1;
            u = u - triu(B);
            %[s,flag,relres,gmres_iter] = gmres(J, -f, [], 1e-6, 40,M);
        end
        %}
        
        % store previous Jacobian
        J1 = J;

        
        %[l,u] = ilu(sparse(J),struct('type','ilutp','milu','row','droptol',1e-3)); %iLU works

        %[l,u] = ilu(sparse(J),struct('type','crout','droptol',1e-3)); %iLU works
        [s,flag,relres,gmres_iter] = gmres(J, -f, [], 1e-6, 40,l,u);
        
        
        %fprintf('---Residual of approximation: %.10f\n',norm(J-l*u));
        %nnz(inv(J))

        %cond(inv(l*u)*J)
        %fprintf('---\n')
        %fprintf('Fill-factor %f\n', nnz(l+u)/nnz(J));
        %fprintf('Ratio: %f\n',nnz(l*u)/nnz(J));
        %[s,flag,relres,gmres_iter] = gmres(J, -f, [], 1e-6, 40,l,u);
        %relres
        %flag
        
        
        assert(flag ~= 1, 'GMRES does not converge!');
        fprintf('     GMRES steps %d\n', gmres_iter(2));
        
        total_step = total_step + gmres_iter(2);
        if (step == 10)
            gmres_step = gmres_step + gmres_iter(2);
        end
        
        %update solution
        x(1:n-1) = x(1:n-1) + s(1:n-1);
        y(1:n-1) = y(1:n-1) + s(n:end);
        
        res = norm(F(x, y, A0, A1, b0, b1, t));
        fprintf('Newton step %d: %.10f\n', iter, res); 
    end
    r(step) = res;
end
fprintf('-----------Data-------------\n');
fprintf('Total GMRES steps used: %d\n',int64(total_step));
fprintf('Average GMRES steps used: %d\n',int64(gmres_step/10));
fprintf('Residual norms: %g\n',res);

%visualize data
semilogy(delta:delta:1,r,'-o');
xlabel('t')
ylabel('residual')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F(x, y, A0, A1, b0, b1, t)
C = (1-t)*A0 + t*A1;
b = (1-t)*b0 + t*b1;
n = length(C);
z = x + 1i*y;
f = diag(z)*(C*conj(z)) - (b(1:n)+1i*b(n+1:end));
f = [real(f); imag(f)];
f(n) = 0; f(end) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = dFdt(x, y, A0, A1)
T = (A1-A0);
n = length(T);
z = x + 1i*y;
f = diag(z)*(T*conj(z));
f = [real(f); imag(f)];
f = [f(1:n-1); f(n+1:end-1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = Jacobian(x, y, A0, A1, t)
C = (1-t)*A0 + t*A1;
n = length(x);
Cx = C*x;
Cy = C*y;
J = [diag(x)*C+diag(Cx)    diag(y)*C+diag(Cy)
     diag(y)*C-diag(Cy)   -diag(x)*C+diag(Cx)];

 %fprintf('Size of Jacobian: %d %d, rank: %d\n', size(J), rank(J));
 %J1 = [J(:,1:n-1) J(:,n+1:end)];
 %fprintf('Drop n th col, size: %d %d, rank: %d\n', size(J1), rank(J1)); 
 %J2 = J(:,1:end-1);
 %fprintf('Drop last col, size: %d %d, rank: %d\n', size(J2), rank(J2)); 
 %J3 = [J(:,1:n-1) J(:,n+1:end-1)];
 %fprintf('Drop both col, size: %d %d, rank: %d\n', size(J3), rank(J3));
 J = [J(1:n-1,1:n-1)      J(1:n-1,n+1:end-1)
      J(n+1:end-1,1:n-1)  J(n+1:end-1,n+1:end-1)];
 
 %J = [J(:,1:n-1) J(:,n+1:end-1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,y] = getsolution(n)
    x = rand(n,1)*2 - 1;
    y = rand(n,1)*2 - 1;
end

function [a, r] = imgs(a)
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = precon_sgs(J)
% upper triangular Symmetric Gauss-Seidel preconditioner
jtj = J'*J;
d = 1 ./ sqrt(diag(jtj));
R = diag(d) * triu(jtj);
end
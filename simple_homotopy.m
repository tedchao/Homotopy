% -----------------------
% Input:
% size of x0: 2n-1 by 1
% size of A1: n by n
% -----------------------
function [J] = simple_homotopy(x0,A1,seed)
format long;

rng(seed);

%forming A0 with same sparsity pattern but different values
%[row,col,val] = find(A1);
%A_e = sparse(row,col,val*(1e-6)*rand(1));
%A_e = full(A_e);
A0 = (1-(1e-6)*rand(1))*A1;

%forming z_start and z_target
x_start = x0 + (1e-4)*rand(size(x0,1),1);
x_target = x_start + (1e-2)*rand(size(x0,1),1);

%get b0 and b1
b0 = p(x_start,A0);
b1 = p(x_target,A1);

%before residual norm 
a = norm(b1-p(x0,A1));

%Homotopy step
ind = 1;
step = zeros(11,1);
for t = 0:0.1:1
    H = (1-t)*p(x0,A0) + t*p(x0,A1);
    b = (1-t)*b0 + t*b1;
    A = (1-t)*A0 + t*A1;
    
    time = 0;
    while norm(H-b) > 1e-12
        %norm(H-b)
        H_0 = (1-t)*p(x0,A0) + t*p(x0,A1); %before
        J = Ja(x0,A);
        [~,R] = qr(J,0);
        s = lsqr(J,-H_0+b,1e-6,10000,R,[]);
        alpha = ls(x0,A,b,s,0.5);
        x0 = x0 + alpha*s;
        H = (1-t)*p(x0,A0) + t*p(x0,A1); %after
        time = time + 1;
        if abs(norm(H_0-b)-norm(H-b)) < 1e-12
            break
        end
    end
    
    H = (1-t)*p(x0,A0) + t*p(x0,A1);
    %norm(H-b)
    %c(ind) = norm(x0);
    f(ind) = norm(H-b);
    step(ind) = time;
    ind = ind + 1;
end

% after residual norm
b = norm(b1-p(x0,A1));

% steps at each homotopy iteration
disp(step);
fprintf('Before residual: %e\n',a);
fprintf('After residual: %e\n',b);

%plot(linspace(0,1,11),f,'-o')
semilogy(linspace(0,1,11),f,'-o')
xlabel('t');
ylabel('residual norm');
size(J)
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Direct line search
% s is direction, l is contraction factor
function a = ls(x,A,b,s,l)
alpha = 1;
while 1
    if norm(p(x+alpha*s,A)-b) <= norm(p(x,A)-b)
        break
    else
        alpha = alpha*l;
    end
end
a = alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Jacobian
function J = Ja(x,A)
n = (size(x,1)+1)/2;
x(2*n) = 0;
%forming J1
for i = 1:n
    for j = 1:n
        J1(i,j) = x(i)*A(i,j);
    end
end
for i = 1:n
    J1(i,i) = J1(i,i) + dot(A(i,:),x(1:n,:));
end

%forming J2
for i = 1:n
    for j = 1:n
        J2(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J2(i,i) = J2(i,i) + dot(A(i,:),x(n+1:2*n,:));
end

%forming J3
for i = 1:n
    for j = 1:n
        J3(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J3(i,i) = J3(i,i) - dot(A(i,:),x(n+1:2*n,:));
end

%forming J4
for i = 1:n
    for j = 1:n
        J4(i,j) = -x(i)*A(i,j);
    end
end
for i = 1:n
    J4(i,i) = J4(i,i) + dot(A(i,:),x(1:n,:));
end

J = [J1 J2; J3 J4];
J = J(:,1:2*n-1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power flow equation 
function out = p(x,A)
%initial guess size 2n-1
n = size(x,1);
real_n = (n+1)/2;

%forming on diagonal
for i = 1:real_n
    Z(i,i) = x(i);
end
for j = real_n+1:n
    Z(j-real_n,j-real_n) = Z(j-real_n,j-real_n) + x(j)*1i;
end

%plug in nonlinear eqns
P = Z*A*conj(diag(Z));

%squeeze out the output
R = real(P);
I = imag(P);

out = zeros(n+1,1);
ind = size(A,1);
for i = 1:ind
    out(i) = R(i);
end
for i = ind+1:n+1
    out(i) = I(i-ind);
end

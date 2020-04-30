function test(x0,A1)

%forming A0 with same sparsity pattern but different values
[row0,col0,val0] = find(A1);
A_e = sparse(row0,col0,val0*(1e-4)*rand(1));
A_e = full(A_e);

A0 = [A1(:,1:99) zeros(100,1)];
%A0 = spones(A1);
norm(A0-A1)
b0 = p(ones(size(x0,1),1),A0);

ind = 1;
%for i = 1:60
%norm(ones(size(x0,1),1)-x0)
%norm(p(x0,A0)-b0)
while norm(p(x0,A0)-b0) > 1e-12
    F = p(x0,A0) - b0;
    f(ind) = norm(F);
    
    J = Ja(x0,A0);
    s = J\(-F);
    
    alpha = ls(x0,A0,b0,s,0.5);
    x0 = x0 + 1*s;

    %norm(F)-norm(p(x0,A0)-b0);
    if abs(norm(F)-norm(p(x0,A0)-b0)) < 1e-12
        break
    end
    
    ind = ind + 1;
    
end
norm(p(x0,A0)-b0)
x0
semilogy(f,'-o')
xlabel('t');
ylabel('residual norm');


% Line search based on backtracking
% s is direction, p is contraction factor
function a = ls(x,A,b,s,l)
alpha = 1;
while 1
    if norm(p(x+alpha*s,A)-b) <= norm(p(x,A)-b + (1e-4)*alpha*Ja(x,A)*s)
        break
    else
        alpha = alpha*l;
    end
end
a = alpha;


% Exact Jacobian matrix
% Jacobian has a form:
% |J1|J2|
% |J3|J4|
% Size of Jacobian is 2nx2n
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
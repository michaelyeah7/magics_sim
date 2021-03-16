function xk1 = StateFunODE(model, xk, uk, Ts)
    %StateFun - Description
    %
    % Syntax: xk1 = StateFun(input)
    %
    % xk: Current states, specified as a column vector of length nx
    % uk: Current inputs, specified as a column vector of length nu
    % Ts: Optional parameters, specified as a comma-separated list (for example p1,p2,p3). 
    %     The same parameters are passed to the prediction model, custom cost function, 
    %     and custom constraint functions of the controller.
    global ip;


    % Get q qdot tau
    q = xk(1 : model.NB, 1);
    qdot = xk(model.NB+1 : model.NB*2, 1);
    tau = model.ST * uk;
    model.tau = tau;

    % Calculate state vector by ODE
    t0 = 0;
    tf = ip.T;
    tspan = [t0, tf];
    options = odeset('RelTol', 1e-3, 'AbsTol', 1e-4, 'Refine', 1, 'Events', @EventsFun);

    x0 = [q; qdot];
    te = t0;
    flag = '';
    while te < tf
        % ODE calculate 
        [t, x, te, xe, ie] = ode45(@(t, x) StandODEFun(t, x, flag), tspan, x0, options);

        % Events occur: Impact dynamics
        if te < tf
            % Get q qdot
            q = xe(1 : model.NB)';
            qdot = xe(model.NB+1 : model.NB*2)';

            % Detect contact
            flag_contact = DetectContact(q, qdot);

            % Impact dynamics
            qdot_impulse = ImpulsiveDynamics(q, qdot, flag_contact);  

            % Update initial state
            x0 = [q; qdot_impulse];
            tspan = [t(end), tf];
        end        
    end

    % Get state vector
    xk1 = x(end, :)';
    
end

function varargout = StandODEFun(t, x, flag)
%StandODEFun - Description
%
% Syntax: varargout = StandODEFun(t, x, flag)
%
% Long description
    switch flag
        case ''
            varargout{1} = DynamicsFun(t, x);
        case 'events'
            [varargout{1:3}] = EventsFun(t, x);
        otherwise
            error(['Error: ODEFun flag ', flag]);
    end
    
end

function Xdot = DynamicsFun(t, X)
%DynamicsFun - Description
%
% Syntax: Xdot = DynamicsFun(t, X)
%
% Dynamics function calculated by ODE 
    global ip;
    global model;

    % Get q qdot tau
    q = X(1 : model.NB, 1);
    qdot = X(model.NB+1 : model.NB*2, 1);
    tau = model.tau;

    % Calcualte H C 
    model.H = CompositeRigidBodyAlgorithm(model, q);
    model.C = InverseDynamics(model, q, qdot, zeros(model.NB, 1));
    model.Hinv = model.H^-1;

    % Calculate contact force in joint space
    flag_contact = DetectContact(q, qdot);
    if sum(flag_contact)~=0 
        % [lambda, fqp, fpd] = SolveContactLCP(q, qdot, tau, flag_contact);
        [lambda, fqp] = CalcContactForceDirect(q, qdot, tau, flag_contact);
    else
        lambda = zeros(model.NB, 1);
    end

    % Forward dynamics
    Tau = tau + lambda;
    qddot = ForwardDynamics(model, q, qdot, Tau);

    % Return Xdot
    Xdot = [qdot; qddot];
end

function [value, isterminal, direction] = EventsFun(t, X)
%EventsFun - Description
%
% Syntax: [value, isterminal, direction] = EventsFun(t, X)
%
% Events function for ODE
    global ip;
    global model;

    % Get q qdot tau
    q = X(1 : model.NB, 1);
    qdot = X(model.NB+1 : model.NB*2, 1);
    tau = model.tau;

    value = ones(model.NC, 1);
    isterminal = ones(model.NC, 1);
    direction = -ones(model.NC, 1);

    % Detect contact
    flag_contact = DetectContact(q, qdot);
    for i=1:model.NC
        if flag_contact(i)==2 % Impact
            % Calculate foot height
        endpos = CalcBodyToBaseCoordinates(model, q, model.idcontact(i), model.contactpoint{i});
        value(i, 1) = endpos(3); 
        end
    end
    
end
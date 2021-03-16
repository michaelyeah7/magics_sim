function JdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact, nf)
%CalcContactJdotQdot - Calculate contact Jdot*Qdot
%
% Syntax: JdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact)
%
% Long description

    k = 0;
    for i=1:model.NC
        if flag_contact(i)~=0
            JdQd = CalcPointAcceleration(model, q, qdot, zeros(model.NB, 1), model.idcontact(i), model.contactpoint{i});

            if nf==2
                JdotQdoti = [JdQd(1, :); JdQd(3, :)]; % only x\z direction
            else 
                if nf==3
                    JdotQdoti = JdQd;
                end
            end

            JdotQdot(k*nf+1:k*nf+nf, :) = JdotQdoti;
            k = k+1;
        end
    end   
end
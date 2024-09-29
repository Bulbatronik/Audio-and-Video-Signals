function mosaic = image_mosaic(im1, im2, H)

        box2 = [1  size(im2,2) size(im2,2)  1 ;
                1  1           size(im2,1)  size(im2,1) ;
                1  1           1            1 ] ;
        % homography 2 -> 1. H is 1 -> 2
        box2_ = H \ box2 ;
        box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
        box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
        ur = min([1, box2_(1,:)]):max([size(im1,2), box2_(1,:)]) ;
        vr = min([1, box2_(2,:)]):max([size(im1,1), box2_(2,:)]) ;

        [u,v] = meshgrid(ur, vr) ;
        im1_ = interp2_multi(im2double(im1), u, v);

        z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
        u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
        v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
        im2_ = interp2_multi(im2double(im2), u_, v_);

        % blending
        mass = ~isnan(im1_) + ~isnan(im2_) ;
        im1_(isnan(im1_)) = 0 ;
        im2_(isnan(im2_)) = 0 ;
        mosaic = (im1_ + im2_) ./ mass ;
        mosaic(mosaic < 0 | mosaic > 1) = 0;

        figure, imagesc(mosaic), axis image off, title('Mosaic');
    end

    % interp2 wrapper for multivalued images
    function new_image = interp2_multi(im, xn, yn)

        new_image = zeros(size(xn, 1), size(xn, 2));
        for chan = 1:size(im, 3)
            new_image(:,:,chan) = interp2(im(:,:,chan), xn, yn, 'cubic');
        end
    end
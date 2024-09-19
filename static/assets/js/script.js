//#######################> Swipper Js Code <##############> 

var swiper = new Swiper(".mySwiper", {
    slidesPerView: "auto",
    spaceBetween: 10,
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
    },
  });
//#######################> Swipper Js Code <##############> 

$(document).ready(function(){
    $('.slider').slick({
        rows: 2,
        arrows: true,     // Enable arrows
        dots: false,
        infinite: true,
        speed: 500,
        slidesToShow: 5,
        slidesToScroll: 1,
        nextArrow: '<button type="button" class="slick-next"><i class="bi bi-caret-right"></i></button>',

        prevArrow: '<button type="button" class="slick-prev"><i class="bi bi-caret-left"></i></button>',

        responsive: [
            {
                breakpoint: 1278,
                settings: {
                    slidesToShow: 4,
                    slidesToScroll: 1,
                    dots: true
                }
            },
            {
                breakpoint: 992,
                settings: {
                    slidesToShow: 3,
                    slidesToScroll: 1,
                    dots: true
                }
            },
            {
                breakpoint: 768,
                settings: {
                    slidesToShow: 2,
                    slidesToScroll: 1,
                    dots: true
                }
            },
            {
                breakpoint: 480,
                settings: {
                    slidesToShow: 1,
                    slidesToScroll: 1
                }
            }
        ]
    });
});




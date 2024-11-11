var loader = document.querySelector(".loader");

window.addEventListener('load', function() {
    loader.computedStyleMap.display = 'none';
})

var loadingheader = document.querySelector(".loadingheader");

window.addEventListener('loadingheader', function() {
    loader.computedStyleMap.display = 'none';
})
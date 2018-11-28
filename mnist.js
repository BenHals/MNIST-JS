paint = false
lines = []
async function runProg(){
    m = document.getElementById('model');
    w = document.getElementById('weight');
    c = document.getElementById('inputCanvas');
    ctx = c.getContext('2d');
    rc = document.getElementById('readCanvas');
    rctx = rc.getContext('2d');
    rctx.drawImage(c, 0, 0, 280, 280, 0, 0, 28, 28);
    p = tf.fromPixels(rc, 1);
    p = p.as4D(1, 28, 28, 1);
    console.log(p.shape);
    // p = p.as2D(1, 28*28);

    model = await  tf.loadModel(tf.io.browserFiles([m.files[0], w.files[0]]));
    model.predict(p).print();
    document.getElementById('output').innerHTML = model.predict(p).argMax(axis=1).get(0).toString();
}

function clearCanv(){
    let canvas = document.getElementById("inputCanvas");
    let ctx = canvas.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 840, 840); 
    lines = [];
}
window.onload = function(){
    let canvas = document.getElementById("inputCanvas");
    let ctx = canvas.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 840, 840);

    canvas.addEventListener('mousedown', function(e){
        paint = true;
        let x = e.clientX - canvas.getBoundingClientRect().left;
        let y = e.clientY - canvas.getBoundingClientRect().top;
        lines.push([]);
        lines[lines.length - 1].push([x, y]);
        // ctx.fillStyle = 'white'
        // ctx.fillRect(x-2, y-2, 4, 4);
        redraw(ctx);
    })
    canvas.addEventListener('mousemove', function(e){
        if(!paint){
            return;
        }
        let x = e.clientX - canvas.getBoundingClientRect().left;
        let y = e.clientY - canvas.getBoundingClientRect().top;
        lines[lines.length - 1].push([x, y]);
        redraw(ctx);
        // ctx.fillStyle = 'white'
        // ctx.fillRect(x-7, y-7, 15, 15);
    })
    canvas.addEventListener('mouseup', function(e){
        paint = false;
        redraw(ctx);
    })
}

function redraw(ctx){
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    for(let l = 0; l < lines.length; l++){
        let points = lines[l];
        ctx.beginPath();
        ctx.moveTo(...points[0]);
        for(let i = 1; i < points.length; i++){
            ctx.lineTo(...points[i])
        }
        ctx.stroke();
    }

}
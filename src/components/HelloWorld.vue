<template>
    <div>
        <h1>TensorFlow.js Object Detection</h1>
        <!-- <select id="base_model">
            <option value="lite_mobilenet_v2">SSD Lite Mobilenet V2</option>
            <option value="mobilenet_v1">SSD Mobilenet v1</option>
            <option value="mobilenet_v2">SSD Mobilenet v2</option>
        </select> -->
        <button type="button" id="run">运行</button>
        <button type="button" id="toggle">切换图片</button>
        <div>
            <img id="image" />
            <canvas id="canvas" width="600" height="399"></canvas>
        </div>
    </div>
</template>

<script>
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'

import * as tf from '@tensorflow/tfjs'
// import { loadGraphModel } from '@tensorflow/tfjs-converter'
import classes from '../mixins/classes'

import imageURL from './image1.jpg'
import image2URL from './image2.jpg'

export default {
    name: 'HelloWorld',
    props: {
        msg: String
    },
    mixins: [classes],
    async mounted() {
        // let modelPromise

        // window.onload = () => (modelPromise = cocoSsd.load())

        class L2 {
            static className = 'L2'

            constructor(config) {
                return tf.regularizers.l1l2(config)
            }
        }
        tf.serialization.registerClass(L2)

        this.model = await tf.loadLayersModel('keras/model.json')
        console.log(this.model)

        const button = document.getElementById('toggle')
        button.onclick = () => {
            image.src = image.src.endsWith(imageURL) ? image2URL : imageURL
        }

        // const select = document.getElementById('base_model')
        // select.onchange = async event => {
        //     const model = await modelPromise
        //     model.dispose()
        //     modelPromise = cocoSsd.load({ base: event.srcElement.options[event.srcElement.selectedIndex].value })
        // }

        const image = document.getElementById('image')
        image.src = imageURL

        const runButton = document.getElementById('run')
        runButton.onclick = async () => {
            // this.model = await tf.loadLayersModel('keras/model.json')
            console.log('model loaded')
            console.time('predict1')

            const result = await this.detect(image)
            console.timeEnd('predict1')

            const c = document.getElementById('canvas')
            const context = c.getContext('2d')
            context.drawImage(image, 0, 0)
            context.font = '10px Arial'

            console.log('number of detections: ', result.length)
            for (let i = 0; i < result.length; i++) {
                context.beginPath()
                context.rect(...result[i].bbox)
                context.lineWidth = 1
                context.strokeStyle = 'green'
                context.fillStyle = 'green'
                context.stroke()
                context.fillText(
                    result[i].score.toFixed(3) + ' ' + result[i].class,
                    result[i].bbox[0],
                    result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10
                )
            }
        }
    },
    methods: {
        detect(img, maxNumBoxes, minScore) {
            if (maxNumBoxes === void 0) {
                maxNumBoxes = 20
            }
            if (minScore === void 0) {
                minScore = 0.5
            }
            return this.infer(img, maxNumBoxes, minScore)
        },
        async infer(img, maxNumBoxes, minScore) {
            const batched = tf.tidy(() => {
                if (!(img instanceof tf.Tensor)) {
                    img = tf.browser.fromPixels(img)
                }
                // Reshape to a single-element batch so we can pass it to executeAsync.
                return tf.expandDims(img)
            })
            const height = batched.shape[1]
            const width = batched.shape[2]

            let INPUT_NODE_NAME = 'input_1'
            let OUTPUT_NODE_NAME = [
                'cls_logits_identity/cls_logits_identity',
                'loc_logits_identity/loc_logits_identity'
            ]
            // let preprocessedInput = tf.div(tf.sub(input.asType('float32'), tf.scalar(255 / 2)), tf.scalar(255 / 2))
            // let reshapedInput = preprocessedInput.reshape([1, ...preprocessedInput.shape])
            let result = await this.model.execute({ [INPUT_NODE_NAME]: batched }, OUTPUT_NODE_NAME)

            // const result = this.model.execute(batched)

            const scores = result[0].dataSync()
            const boxes = result[1].dataSync()

            // clean the webgl tensors
            batched.dispose()
            tf.dispose(result)

            const [maxScores, classes] = this.calculateMaxScores(scores, result[0].shape[1], result[0].shape[2])

            const prevBackend = tf.getBackend()
            // run post process in cpu
            if (tf.getBackend() === 'webgl') {
                tf.setBackend('cpu')
            }
            const indexTensor = tf.tidy(() => {
                const boxes2 = tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[2]])
                return tf.image.nonMaxSuppression(boxes2, maxScores, maxNumBoxes, minScore, minScore)
            })

            const indexes = indexTensor.dataSync()
            indexTensor.dispose()

            // restore previous backend
            if (prevBackend !== tf.getBackend()) {
                tf.setBackend(prevBackend)
            }
            return this.buildDetectedObjects(width, height, boxes, maxScores, indexes, classes)
        },
        buildDetectedObjects(width, height, boxes, scores, indexes, classes) {
            console.log(classes)
            var count = indexes.length
            var objects = []
            for (var i = 0; i < count; i++) {
                var bbox = []
                for (var j = 0; j < 4; j++) {
                    bbox[j] = boxes[indexes[i] * 4 + j]
                }
                var minY = bbox[0] * height
                var minX = bbox[1] * width
                var maxY = bbox[2] * height
                var maxX = bbox[3] * width
                bbox[0] = minX
                bbox[1] = minY
                bbox[2] = maxX - minX
                bbox[3] = maxY - minY
                objects.push({
                    bbox: bbox,
                    class: this.classes[classes[indexes[i]] + 1].displayName,
                    score: scores[indexes[i]]
                })
            }
            console.log(objects)
            return objects
        },
        calculateMaxScores(scores, numBoxes, numClasses) {
            var maxes = []
            var classes = []
            for (var i = 0; i < numBoxes; i++) {
                var max = Number.MIN_VALUE
                var index = -1
                for (var j = 0; j < numClasses; j++) {
                    if (scores[i * numClasses + j] > max) {
                        max = scores[i * numClasses + j]
                        index = j
                    }
                }
                maxes[i] = max
                classes[i] = index
            }
            return [maxes, classes]
        }
    }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped></style>

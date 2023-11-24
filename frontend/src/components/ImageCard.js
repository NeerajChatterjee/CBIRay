import React from 'react'

export default function ImageCard(props) {
    return (
        <div className="card h-auto w-auto d-flex flex-column align-items-center">
            <img src={"http://127.0.0.1:5001/static/chest_xray/" + props.imageFileName}
                height={250}
                width={250}
            />
            <div className="card-body d-flex flex-column align-items-center">
                <div className="card-title fw-semibold">
                    {`${props.imageClassification} [${(props.similarityValue * 100).toFixed(2)}%]`}
                </div>
                <a href={"http://127.0.0.1:5001/static/chest_xray/" + props.imageFileName}
                    target='_blank'
                    class="btn btn-light">View X-Ray
                </a>
            </div>
        </div>
    )
}

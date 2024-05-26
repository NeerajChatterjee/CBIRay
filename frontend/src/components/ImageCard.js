import React from 'react'

export default function ImageCard(props) {
    return (
        <div className="card h-auto w-auto d-flex flex-column align-items-center">
            <img src={"http://127.0.0.1:5001/static/dataset2024/" + props.imageFileName}
                height={100}
                width={100}
            />
            <div className="card-body d-flex flex-column align-items-center">
                <div className="card-title fw-semibold">
                    {`${props.imageClassification}`}
                </div>
                <a href={"http://127.0.0.1:5001/static/dataset2024/" + props.imageFileName}
                    target='_blank'
                    class="btn btn-light">View report
                </a>
            </div>
        </div>
    )
}
